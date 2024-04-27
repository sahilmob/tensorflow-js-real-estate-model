/**
 * Housing CSV data converted to JavaScript arrays for
 * desired inputs and outputs.
 */


// Inputs here represent house size in sqft, and number of bedrooms.
// If you wanted to use more features you could just add sets of 3 or
// more values instead in this 2D Array.
const INPUTS = [
    [3225, 3], [3789, 3], [3636, 3], [3109, 3], [3836, 3], [3510, 3], [3849, 3], [3619, 3], [3619, 3], [3619, 3], [3858, 3], [3858, 3], [3762, 3], [3660, 3], [3510, 3], [3774, 3], [3774, 3], [3774, 3], [3774, 3], [2889, 3], [2889, 3], [3927, 3], [3927, 3], [3348, 3], [3696, 3], [4020, 3], [3849, 3], [3691, 3], [2760, 3], [2760, 3], [2760, 3], [3394, 3], [3666, 3], [3570, 3], [3570, 3], [3732, 3], [3887, 3], [3341, 3], [3394, 3], [3588, 3], [3588, 3], [3588, 3], [3588, 3], [3540, 3], [2858, 3], [2743, 3], [2743, 3], [3966, 3], [3945, 3], [3729, 3], [3780, 3], [3943, 3], [3432, 3], [3432, 3], [3432, 3], [3540, 3], [3726, 3], [3318, 3], [3753, 3], [3753, 3], [3753, 3], [3097, 3], [3675, 3], [3238, 3], [4050, 3], [4050, 3], [3292, 3], [3292, 3], [3292, 3], [3851, 3], [3087, 3], [3087, 3], [3714, 3], [3461, 3], [3461, 3], [3888, 3], [3888, 3], [3693, 3], [4029, 3], [3432, 3], [3525, 3], [3154, 3], [3990, 3], [3638, 3], [3638, 3], [3133, 3], [3936, 3], [3481, 3], [3657, 3], [3660, 3], [3660, 3], [3660, 3], [3354, 3], [3510, 3], [4032, 3], [3588, 3], [3192, 3], [3639, 3], [3639, 3], [3450, 3], [3708, 3], [3708, 3], [3982, 3], [3504, 3], [3504, 3], [3795, 3], [2520, 3], [3797, 3], [2802, 3], [3130, 3], [3588, 3], [3005, 3], [3744, 3], [3169, 3], [3169, 3], [3476, 3], [3476, 3], [2379, 3], [3482, 3], [3936, 3], [3936, 3], [3404, 3], [3404, 3], [3849, 3], [4026, 3], [4026, 3], [3588, 3], [3492, 3], [3492, 3], [4032, 3], [3561, 3], [3573, 3], [3265, 3], [3852, 3], [3588, 3], [3636, 3], [3636, 3], [3063, 3], [3435, 3], [3552, 3], [3055, 3], [3603, 3], [3645, 3], [3852, 3], [3233, 3], [3233, 3], [3240, 3], [3240, 3], [3425, 3], [3425, 3], [3672, 3], [3666, 3], [3744, 3], [2831, 3], [3440, 3], [3440, 3], [3440, 3], [3576, 3], [3118, 3], [3723, 3], [3723, 3], [3132, 3], [3880, 3], [3880, 3], [3880, 3], [3885, 3], [3885, 3], [3885, 3], [2784, 3], [3120, 3], [3375, 3], [3375, 3], [3354, 3], [3501, 3], [3501, 3], [3501, 3], [3244, 3], [2875, 3], [2875, 3], [3096, 3], [3240, 3], [3410, 3], [3510, 3], [3250, 3], [3523, 3], [3523, 3], [2610, 3], [2610, 3], [3202, 3], [2330, 3], [3811, 3], [2964, 3], [3315, 3], [3429, 3], [3744, 3], [4017, 3], [3090, 3], [3023, 3], [3350, 3], [3861, 3], [4040, 3], [3432, 3], [2970, 3], [3492, 3], [3262, 3], [3444, 3], [2960, 3], [3726, 3], [3559, 3], [3105, 3], [3146, 3], [3354, 3], [3354, 3], [3696, 3], [2950, 3], [2950, 3], [2950, 3], [3681, 3], [2940, 3], [3518, 3], [3518, 3], [3518, 3], [3744, 3], [3954, 3], [2784, 3], [2784, 3], [3204, 3], [3899, 3], [3172, 3], [2961, 3], [3912, 3], [3843, 3], [4031, 3], [3365, 3], [3365, 3], [3402, 3], [3458, 3], [3885, 3], [3354, 3], [3354, 3], [3915, 3], [3432, 3], [2873, 3], [2846, 3], [3663, 3], [3663, 3], [3663, 3], [3163, 3], [3461, 3], [3461, 3], [3961, 3], [3024, 3], [3355, 3], [3355, 3], [3355, 3], [3240, 3], [3836, 3], [3273, 3], [3559, 3], [3168, 3], [3726, 3], [3726, 3], [3411, 3], [3149, 3], [3459, 3], [3393, 3], [3393, 3], [4059, 3], [3752, 3], [2888, 3], [3402, 3], [2855, 3], [2855, 3], [3311, 3], [3696, 3], [3588, 3], [2382, 3], [2382, 3], [2382, 3], [2382, 3], [3096, 3], [2985, 3], [3963, 3], [3000, 3], [3666, 3], [2930, 3], [2930, 3], [2930, 3], [2930, 3], [3447, 3], [3895, 3], [3377, 3], [3794, 3], [3492, 3], [3928, 3], [2803, 3], [3361, 3], [3197, 3], [3197, 3], [3038, 3], [3696, 3], [3696, 3], [3582, 3], [3582, 3], [3432, 3], [3270, 3], [3270, 3], [3270, 3], [2846, 3], [2846, 3], [2796, 3], [2796, 3], [3036, 3], [3307, 3], [2803, 3], [2803, 3], [3930, 3], [2292, 3], [3082, 3], [3198, 3], [3325, 3], [3666, 3], [3666, 3], [2988, 3], [2988, 3], [2803, 3], [2803, 3], [2803, 3], [2803, 3], [3798, 3], [3427, 3], [3427, 3], [3517, 3], [3517, 3], [3888, 3], [3888, 3], [3594, 3], [3594, 3], [3076, 3], [4040, 3], [4040, 3], [3714, 3], [3249, 3], [3053, 3], [3306, 3], [3588, 3], [3948, 3], [3542, 3], [3050, 3], [3480, 3], [3888, 3], [4032, 3], [3368, 3], [3392, 3], [3516, 3], [3918, 3], [2330, 3], [3668, 3], [3668, 3], [2371, 3], [3684, 3], [3666, 3], [2716, 3], [2716, 3], [3640, 3], [3203, 3], [3510, 3], [3701, 3], [2707, 3], [3107, 3], [3645, 3], [3645, 3], [3048, 3], [3228, 3], [3228, 3], [3253, 3], [3253, 3], [3566, 3], [2964, 3], [3645, 3], [3826, 3], [3826, 3], [2986, 3], [4050, 3], [3945, 3], [2901, 3], [2901, 3], [3744, 3], [3744, 3], [3795, 3], [3795, 3], [3252, 3], [3888, 3], [3888, 3], [2548, 3], [3474, 3], [3888, 3], [3485, 3], [3485, 3], [2630, 3], [2829, 3], [2730, 3], [2510, 3], [3297, 3], [3996, 3], [3996, 3], [3726, 3], [3726, 3], [3459, 3], [3726, 3], [3859, 3], [3386, 3], [3741, 3], [3501, 3], [2902, 3], [2884, 3], [2884, 3], [3489, 3], [3489, 3], [3450, 3], [3063, 3], [3063, 3], [3864, 3], [3864, 3], [1920, 3], [2550, 3], [3744, 3], [3533, 3], [3922, 3], [3242, 3], [3348, 3], [3483, 3], [2729, 3], [2729, 3], [3969, 3], [3969, 3], [3969, 3], [3969, 3], [3120, 3], [3483, 3], [3676, 3], [3791, 3], [3588, 3], [3900, 3], [3900, 3], [3372, 3], [2599, 3], [3783, 3], [3144, 3], [3477, 3], [3477, 3], [3398, 3], [3015, 3], [3093, 3], [3093, 3], [3339, 3], [2761, 3], [2761, 3], [2761, 3], [2761, 3], [3367, 3], [3222, 3], [3588, 3], [3078, 3], [3102, 3], [3297, 3], [2698, 3], [2947, 3], [2986, 3], [2986, 3], [2473, 3], [3223, 3], [2841, 3], [2841, 3], [4023, 3], [3933, 3], [2748, 3], [3342, 3], [3342, 3], [4057, 3], [3168, 3], [3315, 3], [3804, 3], [3375, 3], [2832, 3], [3150, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [2824, 3], [3354, 3], [3657, 3], [3390, 3], [3951, 3], [3951, 3], [3514, 3], [2904, 3], [2904, 3], [4032, 3], [3693, 3], [3693, 3], [2662, 3], [2662, 3], [3741, 3], [3714, 3], [3735, 3], [2375, 3], [2838, 3], [2838, 3], [2998, 2], [2998, 2], [3230, 2], [2732, 2], [2632, 2], [2732, 2], [2732, 2], [2732, 2], [3899, 2], [2594, 2], [3068, 2], [3623, 2], [2930, 2], [2930, 2], [2396, 2], [2396, 2], [3310, 2], [2624, 2], [2624, 2], [3479, 2], [2884, 2], [2586, 2], [2586, 2], [3120, 2], [2912, 2], [3499, 2], [3580, 2], [3146, 2], [3360, 2], [3138, 2], [3138, 2], [3138, 2], [3138, 2], [3711, 2], [2726, 2], [3392, 2], [3927, 2], [3927, 2], [2990, 2], [2990, 2], [2630, 2], [3672, 2], [2900, 2], [2560, 2], [2560, 2], [2953, 2], [2514, 2], [2514, 2], [2150, 2], [2150, 2], [2334, 2], [3118, 2], [2200, 2], [2520, 2], [3052, 2], [2386, 2], [2632, 2], [2933, 2], [2933, 2], [2524, 2], [2830, 2], [2464, 2], [2686, 2], [2686, 2], [2870, 2], [3548, 2], [2484, 2], [2584, 2], [2584, 2], [3390, 2], [3861, 2], [3714, 2], [3738, 2], [3738, 2], [3375, 2], [2800, 2], [3088, 2], [2313, 2], [2313, 2], [2313, 2], [3024, 2], [3024, 2], [2926, 2], [3912, 2], [3912, 2], [3235, 2], [2344, 2], [2344, 2], [2858, 2], [3024, 2], [2594, 2], [2963, 2], [2856, 2], [3744, 2], [2240, 2], [2240, 2], [2800, 2], [2800, 2], [3080, 2], [2912, 2], [2912, 2], [3136, 2], [2560, 2], [2666, 2], [2666, 2], [2756, 2], [3210, 2], [2258, 2], [2258, 2], [3940, 2], [3940, 2], [2178, 2], [2560, 2], [2708, 2], [3864, 2], [3864, 2], [3118, 2], [3118, 2], [2624, 2], [3508, 2], [3508, 2], [4028, 2], [4028, 2], [3000, 2], [2091, 2], [2552, 2], [2252, 2], [3928, 2], [2162, 2], [2496, 2], [2390, 2], [2801, 2], [2869, 2], [3187, 2], [2734, 2], [2590, 2], [3248, 2], [3248, 2], [3038, 2], [2600, 2], [2600, 2], [2392, 2], [2392, 2], [3369, 2], [2911, 2], [3745, 2], [2625, 2], [2284, 2], [2374, 2], [2680, 2], [2716, 2], [2284, 2], [2600, 2], [2600, 2], [3311, 2], [2480, 2], [2480, 2], [2160, 2], [2160, 2], [2160, 2], [2392, 2], [2442, 2], [3575, 2], [3013, 2], [2562, 2], [3421, 2], [2180, 2], [2524, 2], [2524, 2], [2684, 2], [2732, 2], [2368, 2], [4023, 2], [2392, 2], [2392, 2], [2816, 2], [2990, 2], [2990, 2], [2990, 2], [2990, 2], [2370, 2], [2800, 2], [2800, 2], [3170, 2], [3170, 2], [3600, 2], [3600, 2], [1860, 2], [2760, 2], [2596, 2], [2547, 2], [2218, 2], [2197, 2], [3250, 2], [2188, 2], [3388, 2], [3388, 2], [1780, 2], [2803, 2], [2240, 2], [2971, 2], [2971, 2], [3110, 2], [3110, 2], [3110, 2], [2640, 2], [2236, 2], [3803, 2], [3803, 2], [3803, 2], [2874, 2], [2184, 2], [2415, 2], [3723, 2], [3723, 2], [2421, 2], [2464, 2], [2288, 2], [3068, 2], [2117, 2], [2117, 2], [2160, 2], [2160, 2], [2791, 2], [2960, 2], [2960, 2], [2960, 2], [2370, 2], [3509, 2], [2520, 2], [2240, 2], [2240, 2], [2240, 2], [3443, 2], [2963, 2], [2963, 2], [2460, 2], [1822, 2], [2626, 2], [2626, 2], [3700, 2], [3053, 2], [2112, 2], [2870, 2], [2870, 2], [1939, 2], [2596, 2], [2620, 2], [3055, 2], [2392, 2], [2304, 2], [3399, 2], [3399, 2], [3399, 2], [2870, 2], [2496, 2], [1822, 2], [1822, 2], [3460, 2], [3822, 2], [3822, 2], [2610, 2], [2834, 2], [2834, 2], [2540, 2], [3398, 2], [2560, 2], [2560, 2], [2254, 2], [2375, 2], [2966, 2], [2966, 2], [2338, 2], [2338, 2], [1860, 2], [1860, 2], [2198, 2], [3433, 2], [3433, 2], [2889, 2], [2985, 2], [2009, 2], [1588, 2], [2458, 2], [3429, 2], [2360, 2], [2360, 2], [3517, 2], [2071, 2], [2132, 2], [2500, 2], [2616, 2], [2616, 2], [2373, 2], [2373, 2], [2373, 2], [2165, 2], [2206, 2], [2206, 2], [2980, 2], [3400, 2], [3289, 2], [3289, 2], [3289, 2], [2213, 2], [1594, 2], [1594, 2], [2782, 2], [2782, 2], [2782, 2], [2782, 2], [1812, 2], [1600, 2], [1904, 2], [1904, 2], [1904, 2], [1904, 2], [2458, 2], [1938, 2], [2290, 2], [2290, 2], [2290, 2], [2125, 2], [2131, 2], [2716, 2], [2716, 2], [2543, 2], [1702, 2], [1802, 2], [1802, 2], [2160, 2], [2160, 2], [2750, 2], [2154, 2], [2154, 2], [2154, 2], [2395, 2], [2929, 2], [1938, 2], [1873, 2], [2925, 2], [2160, 2], [2184, 2], [2498, 2], [1992, 2], [2112, 2], [1600, 2], [1760, 2], [1274, 2], [1550, 2], [2208, 2], [2208, 2], [2208, 2], [2208, 2], [2304, 2], [2235, 2], [2235, 2], [1938, 2], [1664, 2], [2612, 2], [2612, 2], [1746, 2], [2423, 2], [2423, 2], [1776, 2], [1776, 2], [2554, 2], [1664, 2], [1664, 2], [2128, 2], [2856, 2], [2090, 2], [2546, 2], [3168, 2], [2616, 2], [2616, 2], [2616, 2], [1771, 2], [2621, 2], [1624, 2], [1584, 2], [2110, 2], [2436, 2], [1664, 2], [1600, 2], [1664, 2], [1536, 2], [1559, 2], [1559, 2], [2508, 2], [2464, 2], [1960, 2], [2256, 2], [1608, 2], [2023, 2], [2274, 2], [1600, 2], [1972, 2], [3009, 2], [2600, 2], [2600, 2], [2014, 2], [2014, 2], [3197, 2], [2084, 2], [1882, 2], [1882, 2], [1882, 2], [2076, 2], [2161, 2], [2161, 2], [2161, 2], [2530, 2], [1662, 1], [1662, 1], [1133, 1], [1456, 1], [1518, 1], [1966, 1], [1243, 1], [2158, 1], [1717, 1], [1902, 1], [1902, 1], [1902, 1], [1790, 1], [1800, 1], [1568, 1], [1944, 1], [1861, 1], [1896, 1], [1896, 1], [1896, 1], [3001, 1], [1611, 1], [1832, 1], [1832, 1], [1832, 1], [3387, 1], [1715, 1], [2655, 1], [1706, 1], [1914, 1], [1930, 1], [1930, 1], [2034, 1], [2034, 1], [2242, 1], [1400, 1], [1516, 1], [1918, 1], [1580, 1], [1580, 1], [2281, 1], [1587, 1], [1587, 1], [1308, 1], [1308, 1], [1752, 1], [1245, 1], [1248, 1], [2470, 1], [2470, 1], [1200, 1], [1148, 1], [1774, 1], [1774, 1], [1774, 1], [1284, 1], [1284, 1], [1760, 1], [1532, 1], [3494, 1], [3494, 1], [1637, 1], [1638, 1], [1914, 1], [1531, 1], [1531, 1], [1754, 1], [809, 1], [1231, 1], [1499, 1], [1499, 1], [1499, 1], [1500, 1], [1500, 1], [1952, 1], [1952, 1], [2037, 1], [1652, 1], [809, 1], [1324, 1], [1834, 1], [1834, 1], [1834, 1], [1776, 1], [894, 1], [1384, 1], [1924, 1], [1536, 1], [1224, 1], [1691, 1], [1485, 1], [1485, 1], [2142, 1], [2142, 1], [2142, 1], [2142, 1], [2164, 1], [1591, 1], [1591, 1], [1835, 1], [1835, 1], [1835, 1], [1835, 1], [809, 1], [805, 1], [1251, 1], [1251, 1], [1251, 1], [1251, 1], [1251, 1], [1532, 1], [1532, 1], [1532, 1], [2082, 1], [1523, 1], [1532, 1], [1532, 1], [1532, 1], [1532, 1], [1568, 1], [1568, 1], [1534, 1], [1600, 1], [1600, 1], [2284, 1], [1750, 1], [1750, 1], [1750, 1], [2105, 1], [2105, 1], [1544, 1], [1841, 1], [1841, 1], [1671, 1], [1567, 1], [1457, 1], [3573, 1], [1596, 1], [1690, 1], [1835, 1], [1955, 1], [1955, 1], [1955, 1], [1795, 1], [1756, 1], [1756, 1], [809, 1], [1479, 1], [1479, 1], [2482, 1], [1322, 1], [1322, 1], [1576, 1], [1576, 1], [1161, 1], [1161, 1], [1161, 1], [1116, 1], [1986, 1], [1188, 1], [1188, 1], [1201, 1], [1559, 1], [1624, 1], [1176, 1], [1566, 1], [1810, 1], [894, 1], [894, 1], [1634, 1], [1634, 1], [809, 1], [1446, 1], [1446, 1], [1488, 1], [1512, 1], [1870, 1], [1870, 1], [1480, 1], [805, 1], [1397, 1], [2114, 1], [2202, 1], [1815, 1], [794, 1], [1547, 1], [1124, 1], [894, 1], [894, 1], [1556, 1], [2514, 1], [2514, 1], [2470, 1], [794, 1], [1374, 1], [1456, 1], [1456, 1], [1528, 1], [1544, 1], [1167, 1], [1592, 1], [1592, 1], [2506, 1], [1094, 1], [1230, 1], [1230, 1], [1120, 1], [1792, 1], [1470, 1], [2230, 1], [2230, 1], [1350, 1], [1350, 1], [1260, 1], [1260, 1], [1613, 1], [894, 1], [1386, 1], [1386, 1], [1470, 1], [960, 1], [960, 1], [1680, 1], [1985, 1], [1441, 1], [2666, 1], [1862, 1], [1380, 1], [1380, 1], [1560, 1], [1192, 1], [1572, 1], [1920, 1], [1920, 1], [1084, 1], [2246, 1], [1536, 1], [1103, 1], [1087, 1], [794, 1], [794, 1], [1474, 1], [1384, 1], [1040, 1], [1040, 1], [1040, 1], [1040, 1], [1605, 1], [2380, 1], [2380, 1], [1075, 1], [1075, 1], [1075, 1], [1075, 1], [1254, 1], [805, 1], [903, 1], [903, 1], [903, 1], [1066, 1], [1363, 1], [1470, 1], [1218, 1], [1667, 1], [794, 1], [1724, 1], [1724, 1], [1249, 1], [1249, 1], [1984, 1], [1224, 1], [1224, 1], [1224, 1], [1262, 1], [1470, 1], [1623, 1], [1352, 1], [1352, 1], [1474, 1], [1474, 1], [1474, 1], [1208, 1], [1510, 1], [1493, 1], [1630, 1], [1630, 1], [1834, 1], [1487, 1], [1096, 1], [1274, 1], [1480, 1], [1776, 1], [1776, 1], [1889, 1], [1503, 1], [1028, 1], [1568, 1], [2482, 1], [1090, 1], [1090, 1], [1658, 1], [1752, 1], [1752, 1], [1390, 1], [1406, 1], [1080, 1], [1258, 1], [1798, 1], [1636, 1], [1094, 1], [1094, 1], [1106, 1], [1500, 1], [1720, 1], [1720, 1], [1720, 1], [1932, 1], [1716, 1], [1344, 1], [1378, 1], [1764, 1], [1356, 1], [1887, 1], [1887, 1], [1887, 1], [1887, 1], [1008, 1], [1470, 1], [1470, 1], [1470, 1], [1470, 1], [1859, 1], [1859, 1], [1282, 1], [1050, 1], [1050, 1], [1050, 1], [1050, 1], [1440, 1], [1568, 1], [928, 1], [1200, 1], [1664, 1], [1664, 1], [1664, 1], [1362, 1], [1518, 1], [1518, 1], [1747, 1], [1747, 1], [1019, 1], [1966, 1], [1568, 1], [1860, 1], [1860, 1], [1179, 1], [1404, 1], [1439, 1], [1096, 1], [1470, 1], [1215, 1], [1215, 1], [1288, 1], [1381, 1], [1773, 1], [1042, 1], [1772, 1], [1512, 1], [1776, 1], [1472, 1], [1706, 1], [1706, 1], [1080, 1], [1478, 1], [899, 1], [899, 1], [1934, 1], [1153, 1], [1558, 1], [1826, 1], [1135, 1], [1260, 1], [1260, 1], [1260, 1], [1024, 1], [1024, 1], [1024, 1], [1024, 1], [1096, 1], [1192, 1], [1990, 1], [1152, 1], [1152, 1], [1048, 1], [1163, 1], [1163, 1], [1896, 1], [1896, 1], [1044, 1], [980, 1], [980, 1], [1056, 1], [1331, 1], [1140, 1], [1192, 1], [1192, 1], [1192, 1], [1192, 1], [1638, 1], [1638, 1], [1638, 1], [1400, 1], [1256, 1], [1340, 1], [1422, 1], [1391, 1], [1626, 1], [1580, 1], [1580, 1], [1273, 1], [1273, 1], [1273, 1], [1664, 1], [1232, 1], [1600, 1], [1694, 1], [1694, 1], [1530, 1], [1530, 1], [1050, 1], [1050, 1], [1456, 1], [1467, 1], [1467, 1], [2558, 1], [2558, 1], [1200, 1], [1836, 1], [1400, 1], [1180, 1], [1248, 1], [1764, 1], [1006, 1], [1688, 1], [1688, 1], [1570, 1], [1655, 1], [1655, 1], [1476, 1], [936, 1], [936, 1], [936, 1], [1249, 1], [1780, 1], [1062, 1], [1404, 1], [1404, 1], [1614, 1], [1015, 1], [1155, 1], [1260, 1], [1888, 1], [2650, 1], [1260, 1], [1040, 1], [1040, 1], [1224, 1], [1224, 1], [1644, 1], [1644, 1], [1644, 1], [1675, 1], [1348, 1], [1344, 1], [1630, 1], [1092, 1], [1092, 1], [1092, 1], [1086, 1], [1086, 1], [1348, 1], [1348, 1], [1176, 1], [2880, 1], [1332, 1], [1632, 1], [2290, 1], [1566, 1], [1075, 1], [1566, 1], [1566, 1], [1566, 1], [1302, 1], [1408, 1], [1532, 1], [1532, 1], [1473, 1], [1048, 1], [1048, 1], [1048, 1], [1048, 1], [1048, 1], [1175, 1], [1107, 1], [1260, 1], [1421, 1], [1137, 1], [900, 1], [1195, 1], [1375, 1], [1375, 1], [1375, 1], [1350, 1], [1350, 1], [1350, 1], [1350, 1], [1575, 1], [1872, 1], [1872, 1], [1928, 1], [1928, 1], [1504, 1], [1620, 1], [1886, 1], [1038, 1], [1312, 1], [1392, 1], [3222, 1], [1344, 1], [1392, 1], [899, 1], [1170, 1], [1344, 1], [1420, 1], [1116, 1], [1116, 1], [1229, 1], [1941, 1], [1660, 1], [1015, 1], [1368, 1], [1368, 1], [1042, 1], [1264, 1], [1264, 1], [1264, 1], [2393, 1], [1716, 1], [1006, 1], [1006, 1], [1110, 1], [1052, 1], [900, 1], [1338, 1], [948, 1], [2031, 1], [900, 1], [1374, 1], [1075, 1], [1530, 1], [1021, 1], [1021, 1], [1044, 1], [1209, 1], [900, 1], [1048, 1], [1048, 1], [1048, 1], [1312, 1], [1392, 1], [1392, 1], [1392, 1], [1428, 1], [1428, 1], [1428, 1], [900, 1], [1078, 1], [1078, 1], [1119, 1], [1119, 1], [1224, 1], [1188, 1], [1232, 1], [1232, 1], [1506, 1], [1085, 1], [1043, 1], [1043, 1], [1043, 1], [973, 1], [1442, 1], [1654, 1], [1654, 1], [2053, 1], [2053, 1], [1318, 1], [1486, 1], [1486, 1], [1275, 1], [1412, 1], [1560, 1], [1386, 1], [1386, 1], [1386, 1], [1192, 1], [1192, 1], [1254, 1], [1344, 1], [1380, 1], [1530, 1], [1530, 1], [1030, 1], [1452, 1], [1482, 1], [1680, 1], [1820, 1], [2158, 1], [2158, 1], [900, 1], [1685, 1], [1110, 1], [1110, 1], [1110, 1], [1156, 1], [1280, 1], [1280, 1], [1241, 1], [1291, 1], [1291, 1], [1305, 1], [1480, 1], [1248, 1], [1051, 1], [900, 1], [1400, 1], [1400, 1], [1102, 1], [1300, 1], [1300, 1], [942, 1], [1776, 1], [1288, 1], [1288, 1], [1342, 1], [1342, 1], [1392, 1], [1291, 1], [1046, 1], [1046, 1], [1046, 1], [1152, 1], [1152, 1], [1152, 1], [1480, 1], [1676, 1], [1833, 1], [900, 1], [1450, 1], [1450, 1], [1042, 1], [1042, 1], [1103, 1], [1103, 1], [1255, 1], [1094, 1], [1164, 1], [1232, 1], [1232, 1], [1470, 1], [1666, 1], [1666, 1], [1467, 1], [1384, 1], [1329, 1], [1120, 1], [1698, 1], [1015, 1], [1015, 1], [1015, 1], [1120, 1], [1120, 1], [965, 1], [1015, 1], [1015, 1], [1015, 1], [1075, 1], [1536, 1], [1536, 1], [1008, 1], [1244, 1], [1244, 1], [1382, 1], [1382, 1], [1382, 1], [1039, 1], [1513, 1], [1513, 1], [1048, 1], [1152, 1], [900, 1], [1344, 1], [1367, 1], [1007, 1], [1096, 1], [1322, 1], [1322, 1], [1322, 1], [720, 1], [1344, 1], [1574, 1], [1008, 1], [1008, 1], [546, 1], [1428, 1], [720, 1], [768, 1], [1226, 1], [1296, 1], [1632, 1], [1126, 1], [1200, 1], [1849, 1], [900, 1], [1380, 1], [1148, 1], [2218, 1], [1013, 1], [1068, 1], [1068, 1], [2607, 1], [900, 1], [1370, 1], [1180, 1], [974, 1], [1167, 1], [1167, 1], [899, 1], [1408, 1], [900, 1], [900, 1], [930, 1], [1173, 1], [964, 1], [1376, 1], [1376, 1], [1338, 1], [973, 1], [1079, 1], [1079, 1], [881, 1], [1050, 1], [1050, 1], [900, 1], [900, 1], [900, 1], [900, 1], [1350, 1], [1075, 1], [1008, 1], [1376, 1], [1376, 1], [1056, 1], [1056, 1], [1056, 1], [1586, 1], [1392, 1], [1285, 1], [1285, 1], [1368, 1], [975, 1], [900, 1], [1599, 1], [1599, 1], [900, 1], [900, 1], [1096, 1], [1273, 1], [1408, 1], [546, 1], [1024, 1], [1074, 1], [1111, 1], [900, 1], [1092, 1], [1248, 1], [1600, 1], [1831, 1], [1831, 1], [1831, 1], [2105, 1], [720, 1], [767, 1], [974, 1], [974, 1], [3904, 1], [1042, 1], [1042, 1], [1075, 1], [3001, 1], [988, 1], [988, 1], [1075, 1], [1350, 1], [1350, 1], [1350, 1], [1883, 1], [1123, 1], [1123, 1], [1263, 1], [1263, 1], [1368, 1], [801, 1], [801, 1], [1664, 1], [900, 1], [912, 1], [912, 1], [1239, 1], [1368, 1], [1368, 1], [1008, 1], [1384, 1], [1384, 1], [1056, 1], [1056, 1], [1056, 1], [1117, 1], [1152, 1], [720, 1], [720, 1], [1075, 1], [1075, 1], [1075, 1], [1075, 1], [1075, 1], [1036, 1], [1036, 1], [899, 1], [1210, 1], [1210, 1], [1940, 1], [900, 1], [1394, 1], [1092, 1], [1260, 1], [1330, 1], [900, 1], [2399, 1], [1200, 1], [935, 1], [546, 1], [1188, 1], [2079, 1], [720, 1], [1397, 1], [1230, 1], [1242, 1], [1242, 1], [1390, 1], [2531, 1], [1042, 1], [1536, 1], [1536, 1], [1536, 1], [1684, 1], [1486, 1], [2414, 1], [2414, 1], [2414, 1], [1198, 1], [1198, 1], [1198, 1], [926, 1], [1048, 1], [1048, 1], [1048, 1], [1075, 1], [1075, 1], [1184, 1], [1298, 1], [1298, 1], [1298, 1], [1875, 1], [1104, 1], [1161, 1], [1153, 1], [1245, 1], [1152, 1], [1585, 1], [1315, 1], [1315, 1], [1171, 1], [1446, 1], [1972, 1], [672, 1], [900, 1], [1218, 1], [1218, 1], [1555, 1], [978, 1], [1015, 1], [1015, 1], [1170, 1], [1560, 1], [1008, 1], [912, 1], [667, 1], [667, 1], [975, 1], [1015, 1], [1144, 1], [1191, 1], [1682, 1], [1100, 1], [972, 1], [972, 1], [972, 1], [1442, 1], [1442, 1], [1434, 1], [1434, 1], [1120, 1], [1120, 1], [1120, 1], [1620, 1], [900, 1], [900, 1], [903, 1], [1153, 1], [1153, 1], [1120, 1], [1120, 1], [1120, 1], [1656, 1], [1656, 1], [1656, 1], [1050, 1], [840, 1], [1008, 1], [1008, 1], [1300, 1], [1300, 1], [1568, 1], [864, 1], [1328, 1], [1328, 1], [900, 1], [1075, 1], [1190, 1], [1360, 1], [1157, 1], [1190, 1], [1382, 1], [1364, 1], [1364, 1], [1364, 1], [1388, 1], [1651, 1], [2022, 1], [1190, 1], [1260, 1], [1075, 1], [1260, 1], [1050, 1], [983, 1], [983, 1], [983, 1], [986, 1], [986, 1], [1176, 1], [864, 1], [1182, 1], [1131, 1], [1152, 1], [1224, 1], [1096, 1], [899, 1], [1330, 1], [1330, 1], [1096, 1], [433, 1], [1024, 1], [1519, 1], [1494, 1], [958, 1], [433, 1], [1336, 1], [1075, 1], [1498, 1], [1092, 1], [1181, 1], [1181, 1], [1008, 1], [1112, 1], [1112, 1], [1326, 1], [1326, 1], [1326, 1], [1326, 1], [864, 1], [864, 1], [864, 1], [1381, 1], [1075, 1], [2058, 1], [1445, 1], [1520, 1], [992, 1], [1116, 1], [1116, 1], [1694, 1], [1386, 1], [1174, 1], [1430, 1], [950, 1], [1204, 1], [1204, 1], [1204, 1], [1610, 1], [1610, 1], [1610, 1], [1881, 1], [1024, 1], [900, 1], [900, 1], [1050, 1], [1164, 1], [1164, 1], [2525, 1], [1440, 1], [1440, 1], [952, 1], [1156, 1], [1156, 1], [1156, 1], [1156, 1], [2482, 1], [1344, 1], [1344, 1], [1344, 1], [1881, 1], [675, 1], [2307, 1], [1228, 1], [675, 1], [900, 1], [919, 1], [1006, 1], [1006, 1], [1241, 1], [1881, 1], [1350, 1], [1350, 1], [1350, 1], [1529, 0], [2600, 0], [1271, 0], [1990, 0], [1990, 0], [1792, 0], [1792, 0], [1792, 0], [1234, 0], [1680, 0], [714, 0], [2362, 0], [2362, 0], [2362, 0], [1154, 0], [1211, 0], [3953, 0], [2580, 0], [1650, 0], [1436, 0], [1139, 0], [1139, 0], [3887, 0], [697, 0], [1659, 0], [1365, 0], [3637, 0], [2778, 0], [1052, 0], [1784, 0], [1784, 0], [1031, 0], [1096, 0], [832, 0], [536, 0], [522, 0], [1492, 0], [2587, 0], [1214, 0], [1958, 0], [1956, 0], [1737, 0], [1701, 0], [1787, 0], [1212, 0], [1634, 0], [647, 0], [2287, 0], [2266, 0], [2301, 0], [1148, 0], [1197, 0], [3926, 0], [2535, 0], [1582, 0], [1369, 0], [1097, 0], [1092, 0], [3848, 0], [646, 0], [1592, 0], [1325, 0], [3594, 0], [2697, 0], [1000, 0], [1759, 0], [1727, 0], [953, 0], [1080, 0], [819, 0], [460, 0], [432, 0], [595, 0]
];

// Outputs here represent sale price of the house.
const OUTPUTS = [262300, 270000, 262000, 259700, 259000, 258800, 258600, 258500, 258500, 258500, 258400, 258400, 258200, 257400, 257200, 257100, 257100, 257100, 257100, 255700, 255700, 255600, 255600, 255500, 254600, 254400, 254200, 254100, 253800, 253800, 253800, 253600, 253200, 252700, 252700, 252600, 252600, 252500, 252100, 252100, 252100, 252100, 252100, 252000, 251700, 251600, 251600, 251400, 251300, 250600, 250600, 250200, 249600, 249600, 249600, 249400, 248900, 248400, 248200, 248200, 248200, 247200, 247200, 247000, 246900, 246900, 246700, 246700, 246700, 246500, 245600, 245600, 245000, 244800, 244800, 244200, 244200, 242600, 242500, 242300, 242300, 242100, 242100, 241800, 241800, 241700, 241200, 240900, 240900, 240900, 240900, 240900, 240700, 240700, 240700, 240600, 240500, 240500, 240500, 240000, 239900, 239900, 239800, 239500, 239500, 239500, 239300, 239200, 239100, 238900, 238900, 238800, 238800, 238600, 238600, 238600, 238600, 238500, 238200, 238200, 238000, 237800, 237800, 237700, 237700, 237700, 237100, 236800, 236800, 236700, 236500, 236500, 236200, 236000, 235800, 235600, 235600, 235500, 235500, 235400, 235200, 235100, 235100, 235000, 234600, 234600, 234400, 234400, 234400, 234400, 234400, 234300, 233900, 233600, 233600, 233600, 233600, 233600, 233400, 232900, 232900, 232100, 232100, 232100, 232100, 231900, 231900, 231900, 231700, 231500, 231500, 231500, 231400, 231300, 231300, 231300, 230900, 230800, 230800, 230700, 230600, 230600, 230100, 230000, 230000, 230000, 229900, 229900, 229800, 229300, 229300, 229100, 229000, 228900, 228800, 228700, 228600, 228200, 228200, 228200, 228200, 228100, 228000, 227900, 227700, 227700, 227300, 227300, 227200, 227100, 227100, 227100, 227100, 227000, 226800, 226800, 226800, 226600, 226500, 226400, 226400, 226400, 226400, 226300, 225800, 225800, 225800, 225700, 225600, 225400, 225300, 225100, 225000, 224900, 224900, 224400, 224200, 224200, 223700, 223700, 223700, 223500, 223000, 222900, 222400, 222400, 222400, 222200, 222200, 222200, 222200, 222100, 222100, 222100, 222100, 222000, 221900, 221800, 221800, 221700, 221700, 221700, 221600, 221400, 221400, 221200, 221200, 221100, 221000, 220800, 220800, 220700, 220700, 220300, 220300, 220000, 219900, 219900, 219900, 219900, 219900, 219700, 219700, 219600, 219600, 219500, 219500, 219500, 219500, 219400, 219400, 219000, 219000, 218900, 218900, 218700, 218500, 218300, 218300, 218200, 218200, 218200, 218100, 218100, 218000, 217900, 217900, 217900, 217400, 217400, 217300, 217300, 217300, 217200, 217000, 217000, 217000, 216600, 216600, 216600, 216200, 216200, 216200, 216100, 216100, 215900, 215900, 215900, 215900, 215900, 215600, 215600, 215300, 215300, 214800, 214800, 214700, 214700, 214200, 214200, 214200, 214000, 213900, 213300, 213200, 213200, 213000, 212800, 212700, 212600, 212600, 212500, 212200, 212200, 212200, 212200, 212100, 212100, 212100, 212000, 211700, 211600, 211500, 211500, 211400, 211300, 211000, 210900, 210800, 210700, 210700, 210700, 210600, 210500, 210500, 210500, 210500, 210500, 210300, 210300, 210300, 210300, 210200, 210200, 209800, 209200, 209200, 208700, 208700, 208500, 208500, 207900, 207800, 207800, 207400, 207200, 207200, 207000, 207000, 206800, 206700, 206500, 206200, 206100, 205800, 205800, 205600, 205600, 205500, 205500, 205300, 205100, 205000, 204900, 204300, 204000, 204000, 204000, 204000, 203800, 203500, 203500, 203300, 203300, 203200, 203000, 203000, 202900, 202500, 202400, 202300, 202300, 202000, 202000, 202000, 202000, 202000, 202000, 201100, 201100, 200900, 200400, 200300, 200200, 200200, 200000, 199600, 199500, 198400, 198300, 198300, 198100, 197900, 197700, 197700, 140000, 197300, 197300, 197300, 197300, 197200, 196900, 196700, 196600, 172000, 145000, 150000, 195100, 195100, 195100, 194800, 194200, 193700, 193700, 193200, 192900, 192400, 192100, 192100, 192100, 191700, 190800, 190600, 190300, 190200, 190200, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 190100, 189800, 189700, 189700, 189700, 189600, 189000, 189000, 188700, 188500, 188500, 186900, 186900, 186700, 186300, 185100, 184400, 183500, 183500, 250000, 210800, 210700, 235000, 209400, 209400, 222000, 209300, 208300, 208100, 208100, 207900, 207800, 207800, 207500, 207500, 207400, 207300, 207300, 207200, 206000, 205400, 205400, 205200, 205000, 205000, 204900, 204800, 204800, 203900, 203900, 203900, 203900, 203900, 202800, 202600, 202400, 202400, 202000, 202000, 201900, 201900, 201300, 200800, 200800, 200800, 200700, 200700, 200500, 200500, 200500, 200400, 200300, 200200, 199900, 199500, 199300, 199200, 199200, 198300, 197800, 197600, 197200, 197200, 196700, 196500, 196400, 196200, 196200, 196000, 195900, 195800, 195800, 195800, 195400, 195300, 195000, 194300, 194300, 194300, 193400, 193400, 193100, 192600, 192600, 192300, 192100, 192100, 191700, 191500, 191200, 191200, 190900, 190600, 190100, 190100, 189600, 189600, 189600, 189500, 189500, 189500, 188900, 188900, 188900, 188900, 188800, 188700, 188700, 188700, 188700, 188600, 188600, 188500, 188500, 188500, 188400, 188400, 188200, 187100, 187100, 186600, 186600, 186100, 186000, 185800, 185600, 185500, 185400, 185400, 185300, 185300, 185300, 185200, 185100, 184900, 183700, 183700, 183200, 183100, 183100, 182500, 182200, 181900, 181800, 181800, 181500, 181400, 181200, 180600, 180600, 179700, 179500, 179500, 179500, 178300, 178300, 178200, 178200, 178200, 178200, 178000, 177400, 177200, 177000, 176900, 176700, 176700, 176700, 176600, 176500, 176300, 175900, 175700, 175700, 175700, 175100, 175100, 175100, 175100, 175000, 174500, 174500, 174100, 174100, 173300, 173300, 173200, 172700, 172300, 172100, 171900, 171800, 171000, 170900, 170900, 170900, 170800, 170800, 170100, 170000, 170000, 169100, 169100, 169100, 168900, 168300, 168200, 168200, 168200, 167900, 167500, 167400, 167400, 167400, 167300, 167200, 167000, 166800, 166500, 166500, 166500, 166500, 166500, 166500, 166500, 166500, 166400, 166300, 165700, 165500, 165500, 165500, 165300, 165000, 165000, 164900, 164700, 164600, 164600, 164500, 164100, 163700, 163700, 163700, 163500, 163200, 163100, 163000, 162900, 162800, 162800, 162800, 162800, 162600, 162300, 162000, 162000, 161900, 161800, 161800, 161000, 160700, 160700, 160600, 160500, 160400, 160400, 160300, 160300, 159800, 159800, 159300, 159300, 158500, 158500, 158500, 157900, 157900, 157800, 157500, 157300, 157200, 157200, 157100, 157000, 157000, 156400, 156300, 156100, 156000, 156000, 156000, 155800, 155800, 155800, 155700, 155700, 155700, 155600, 155500, 155200, 155200, 155200, 154700, 154500, 154500, 154100, 154100, 154100, 154100, 153500, 153400, 152900, 152900, 152900, 152900, 152900, 152700, 152600, 152600, 152600, 152000, 151900, 151900, 151900, 151000, 150900, 150700, 150700, 150300, 150300, 150200, 150100, 150100, 150100, 150000, 150000, 149600, 148400, 148000, 146800, 146700, 146400, 146200, 145800, 145700, 145600, 145400, 145400, 144800, 144800, 144800, 144800, 144100, 144000, 144000, 143900, 143200, 143000, 143000, 142800, 142700, 142700, 142500, 142500, 142500, 142200, 142200, 141800, 141700, 141600, 141300, 140100, 139500, 139500, 139500, 139300, 139200, 138800, 138700, 138000, 138000, 137700, 137500, 136700, 136200, 136100, 136100, 136000, 135900, 135800, 135700, 135400, 135200, 135200, 134500, 133300, 133000, 132200, 132200, 131900, 131900, 111000, 131800, 131300, 120000, 75000, 131000, 130300, 130300, 130300, 100000, 180000, 195000, 185000, 174000, 179000, 172900, 172700, 172500, 172400, 171900, 171900, 171900, 171800, 171200, 170800, 170300, 169900, 169700, 169700, 169700, 169300, 168800, 168700, 168700, 168700, 168500, 168200, 167700, 167600, 167200, 166300, 166300, 166000, 166000, 165400, 164400, 163600, 163600, 163400, 163400, 163300, 163200, 163200, 163100, 163100, 162900, 162600, 162600, 162300, 162300, 162200, 162000, 162000, 162000, 162000, 161900, 161900, 161100, 160900, 160700, 160700, 160400, 160300, 160300, 160000, 160000, 159700, 159500, 159500, 159500, 159500, 159500, 159500, 159500, 159400, 159400, 159400, 158900, 158800, 158800, 158600, 158600, 158600, 158500, 158400, 158400, 158400, 158200, 158100, 158100, 158000, 158000, 157800, 157800, 157800, 157800, 157800, 157500, 157500, 157300, 157300, 157300, 157300, 156600, 156400, 156100, 156100, 156100, 156100, 156100, 156000, 156000, 156000, 155800, 155600, 155600, 155600, 155600, 155600, 155600, 155600, 155200, 155200, 155200, 155200, 154800, 154800, 154800, 154800, 154800, 154600, 154500, 154500, 154400, 154000, 153900, 153700, 153600, 153600, 153600, 153200, 153200, 153200, 152800, 152700, 152700, 152300, 152200, 152200, 152200, 151900, 151900, 151900, 151900, 151700, 151700, 151700, 151600, 151500, 151400, 151400, 151400, 151300, 151200, 151100, 151100, 151100, 151000, 151000, 151000, 151000, 150800, 150800, 150800, 150800, 150800, 150800, 150800, 150700, 150600, 150600, 150600, 150600, 150400, 150200, 150200, 150100, 149600, 149600, 149600, 149600, 149600, 149400, 149300, 149300, 149300, 149300, 149300, 149300, 149100, 149000, 149000, 149000, 148800, 148800, 148800, 148700, 148600, 148500, 148500, 148500, 148300, 148300, 148200, 148200, 148200, 148100, 148100, 148100, 148000, 147900, 147900, 147900, 147900, 147700, 147700, 147500, 147400, 147400, 147400, 147000, 146700, 146600, 146600, 146500, 146400, 146200, 146100, 146000, 145900, 145900, 145900, 145800, 145700, 145700, 145700, 145700, 145700, 145700, 145700, 145000, 145000, 145000, 145000, 145000, 144900, 144900, 144900, 144900, 144900, 144800, 144800, 144700, 144600, 144500, 144500, 144500, 144400, 144400, 144300, 144200, 144200, 144200, 144200, 144100, 144100, 144000, 144000, 143700, 143700, 143700, 143400, 143400, 143300, 143200, 143200, 143200, 143100, 143000, 143000, 143000, 143000, 143000, 143000, 142900, 142800, 142800, 142800, 142600, 142600, 142500, 142500, 142500, 142400, 142400, 142300, 142000, 142000, 141900, 141800, 141800, 141600, 141600, 141600, 141600, 141600, 141600, 141500, 141400, 141300, 141300, 140700, 140700, 140700, 140700, 140700, 140500, 140500, 140500, 140500, 140500, 140500, 140500, 140400, 140300, 140300, 140300, 140300, 140300, 140300, 140100, 140100, 140100, 140100, 140100, 140000, 140000, 140000, 140000, 140000, 139900, 139900, 139800, 139800, 139800, 139700, 139700, 139700, 139600, 139500, 139400, 139400, 139400, 139400, 139400, 139300, 139300, 139000, 139000, 138900, 138900, 138900, 138800, 138800, 138600, 138600, 138600, 138200, 138200, 138200, 138100, 138100, 138100, 138100, 138000, 138000, 138000, 138000, 137900, 137900, 137700, 137600, 137600, 137500, 137500, 137500, 137400, 137400, 137300, 137200, 137200, 137100, 137000, 136900, 136900, 136900, 136900, 136900, 136900, 136900, 136900, 136800, 136700, 136600, 136600, 136500, 136500, 136400, 136400, 136200, 136200, 136200, 136100, 136000, 136000, 136000, 136000, 135800, 135800, 135700, 135700, 135700, 135700, 135700, 135600, 135600, 135500, 135500, 135400, 135300, 135100, 135100, 135000, 135000, 135000, 134900, 134700, 134700, 134600, 134500, 134500, 134500, 134500, 134500, 134200, 134200, 134200, 134200, 134000, 134000, 134000, 134000, 134000, 133900, 133800, 133800, 133800, 133800, 133800, 133800, 133800, 133800, 133700, 133500, 133500, 133400, 133100, 133100, 132900, 132900, 132700, 132700, 132600, 132500, 132300, 132300, 132300, 132100, 132000, 132000, 132000, 132000, 131900, 131900, 131900, 131900, 131800, 131700, 131700, 131700, 131700, 131700, 131700, 131600, 131600, 131600, 131500, 131400, 131400, 131400, 131400, 131400, 131200, 131200, 131200, 131200, 131200, 131100, 131100, 131100, 131100, 131000, 131000, 131000, 130900, 130900, 130900, 130900, 130800, 130700, 130600, 130600, 130500, 130500, 130400, 130400, 130400, 130400, 130300, 130200, 130200, 130200, 130100, 130100, 130100, 130100, 130100, 130000, 129900, 129900, 129900, 129700, 129600, 129600, 129400, 129400, 129100, 129100, 128900, 128900, 128700, 128700, 128500, 128300, 128200, 128200, 128200, 128200, 128200, 128100, 128100, 128100, 128100, 128100, 128100, 128000, 127900, 127900, 127900, 127900, 127900, 127800, 127800, 127800, 127700, 127600, 127500, 127500, 127500, 127400, 127400, 127400, 127400, 127400, 127400, 127300, 127300, 127300, 127200, 127200, 127100, 127000, 127000, 127000, 126900, 126900, 126900, 126900, 126900, 126800, 126800, 126700, 126700, 126600, 126600, 126600, 126600, 126600, 126500, 126500, 126300, 126300, 126300, 126300, 126200, 126200, 126100, 126100, 126100, 126000, 126000, 125900, 125800, 125700, 125600, 125600, 125500, 125400, 125400, 125200, 125200, 125100, 125100, 125100, 125100, 125100, 125000, 124900, 124900, 124900, 124500, 124500, 124500, 124500, 124500, 124200, 124100, 124100, 124100, 124000, 124000, 124000, 124000, 124000, 123900, 123900, 123700, 123700, 123700, 123700, 123700, 123500, 123100, 123000, 122800, 122800, 122500, 122500, 122500, 122500, 122500, 122400, 122400, 122400, 122400, 122300, 122300, 122300, 122100, 122100, 122100, 122100, 122100, 122100, 122000, 122000, 122000, 121800, 121600, 121400, 121200, 121100, 121000, 121000, 121000, 121000, 121000, 120900, 120900, 120800, 120700, 120700, 120600, 120600, 120500, 120300, 120300, 120300, 120200, 120100, 120100, 120100, 119900, 119900, 119600, 119500, 119400, 119300, 119300, 119100, 119000, 118900, 118800, 118700, 118700, 118700, 118600, 118600, 118500, 118500, 118400, 118200, 118100, 118100, 118100, 118000, 117900, 117900, 117900, 117800, 117700, 117700, 117600, 117200, 117200, 117200, 117200, 117100, 117000, 117000, 117000, 116900, 116900, 116900, 116700, 116600, 116500, 116500, 116500, 116400, 116300, 116300, 116300, 116200, 116200, 116200, 116200, 116200, 116100, 116100, 116100, 116100, 116000, 116000, 115900, 115900, 115800, 115800, 115800, 115800, 115700, 115700, 115700, 115700, 115600, 115500, 115500, 115500, 115500, 115400, 115400, 115400, 115300, 115300, 115300, 115300, 115100, 115100, 115100, 115100, 115100, 115000, 115000, 115000, 114900, 114800, 114800, 114800, 114800, 114800, 114700, 114700, 114700, 114600, 114600, 114600, 114300, 114200, 114000, 114000, 113600, 113600, 113600, 113600, 113600, 113400, 113400, 113200, 113100, 113100, 113100, 112900, 112800, 112300, 112300, 112300, 112100, 112000, 111800, 111700, 111600, 111600, 111500, 111400, 111300, 111100, 111000, 111000, 110900, 110800, 110700, 110600, 110600, 110600, 110600, 109900, 109900, 109900, 109900, 109700, 109700, 109700, 109500, 109400, 109400, 109400, 109400, 109400, 109300, 109000, 109000, 109000, 108900, 108800, 108800, 108600, 108600, 108100, 108000, 107900, 107900, 107800, 107800, 107700, 107400, 107400, 107400, 107400, 107300, 107200, 107200, 107200, 107200, 106900, 106700, 106600, 106400, 106400, 106200, 106200, 106200, 106200, 106200, 106100, 105900, 105900, 105900, 105900, 105900, 105700, 105700, 105600, 105600, 105600, 105500, 105400, 105400, 105000, 105000, 105000, 104900, 104900, 104900, 104900, 104900, 104900, 104800, 104500, 104500, 104500, 104400, 104400, 104400, 104300, 104000, 104000, 103800, 103800, 103800, 103800, 103700, 103400, 103200, 103000, 103000, 103000, 103000, 102700, 102700, 102600, 102500, 102100, 102100, 101700, 101500, 101500, 101500, 101400, 101400, 101400, 101300, 100100, 99900, 99900, 99600, 99400, 99300, 99100, 99100, 99000, 98900, 98900, 98800, 98300, 98200, 98000, 98000, 97800, 97600, 97400, 97300, 97300, 97100, 96700, 96700, 96600, 96600, 96600, 96600, 96400, 96400, 96400, 96000, 95200, 95100, 95000, 94700, 94300, 94200, 94200, 93600, 93300, 93100, 92800, 92600, 92300, 92300, 92300, 92300, 92300, 92300, 91700, 91500, 91400, 91400, 91400, 91100, 91100, 91100, 90900, 90900, 90300, 90100, 90100, 90100, 90100, 90000, 89800, 89800, 89800, 89600, 89200, 89200, 88500, 88300, 87100, 87100, 86700, 86700, 86300, 67000, 75000, 70000, 60000, 127700, 126600, 117200, 115100, 115100, 113700, 113700, 113700, 103100, 101400, 99400, 97100, 97100, 97100, 96400, 96300, 91900, 87000, 86900, 82100, 78800, 78800, 78800, 74300, 72200, 70900, 70100, 64600, 61600, 53900, 53900, 52200, 47700, 46800, 44800, 43600, 137667, 129343, 126735, 116717, 122654, 119598, 118594, 122823, 112819, 104765, 108896, 104448, 100437, 106377, 96794, 96827, 96947, 92889, 88445, 83879, 87069, 80764, 85231, 75751, 79712, 71678, 73940, 66959, 65625, 60733, 63317, 56487, 52692, 50366, 46867, 46517, 48921];


export const TRAINING_DATA = {
  inputs: INPUTS,
  outputs: OUTPUTS
};
