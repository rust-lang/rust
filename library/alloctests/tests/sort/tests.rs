use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::panic::{self, AssertUnwindSafe};
use std::rc::Rc;
use std::{env, fs};

use crate::sort::ffi_types::{F128, FFIOneKibiByte};
use crate::sort::{Sort, known_good_stable_sort, patterns};

#[cfg(miri)]
const TEST_LENGTHS: &[usize] = &[2, 3, 4, 7, 10, 15, 20, 24, 33, 50, 100, 171, 300];

// node.js gives out of memory error to use with length 1_100_000
#[cfg(all(not(miri), target_os = "emscripten"))]
const TEST_LENGTHS: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 20, 24, 30, 32, 33, 35, 50, 100, 200, 500, 1_000,
    2_048, 5_000, 10_000, 100_000,
];

#[cfg(all(not(miri), not(target_os = "emscripten")))]
const TEST_LENGTHS: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 20, 24, 30, 32, 33, 35, 50, 100, 200, 500, 1_000,
    2_048, 5_000, 10_000, 100_000, 1_100_000,
];

fn check_is_sorted<T: Ord + Clone + Debug, S: Sort>(v: &mut [T]) {
    let seed = patterns::get_or_init_rand_seed();

    let is_small_test = v.len() <= 100;
    let v_orig = v.to_vec();

    <S as Sort>::sort(v);

    assert_eq!(v.len(), v_orig.len());

    for window in v.windows(2) {
        if window[0] > window[1] {
            let mut known_good_sorted_vec = v_orig.clone();
            known_good_stable_sort::sort(known_good_sorted_vec.as_mut_slice());

            if is_small_test {
                eprintln!("Original: {:?}", v_orig);
                eprintln!("Expected: {:?}", known_good_sorted_vec);
                eprintln!("Got:      {:?}", v);
            } else {
                if env::var("WRITE_LARGE_FAILURE").is_ok() {
                    // Large arrays output them as files.
                    let original_name = format!("original_{}.txt", seed);
                    let std_name = format!("known_good_sorted_{}.txt", seed);
                    let testsort_name = format!("{}_sorted_{}.txt", S::name(), seed);

                    fs::write(&original_name, format!("{:?}", v_orig)).unwrap();
                    fs::write(&std_name, format!("{:?}", known_good_sorted_vec)).unwrap();
                    fs::write(&testsort_name, format!("{:?}", v)).unwrap();

                    eprintln!(
                        "Failed comparison, see files {original_name}, {std_name}, and {testsort_name}"
                    );
                } else {
                    eprintln!(
                        "Failed comparison, re-run with WRITE_LARGE_FAILURE env var set, to get output."
                    );
                }
            }

            panic!("Test assertion failed!")
        }
    }
}

fn test_is_sorted<T: Ord + Clone + Debug, S: Sort>(
    test_len: usize,
    map_fn: impl Fn(i32) -> T,
    pattern_fn: impl Fn(usize) -> Vec<i32>,
) {
    let mut test_data: Vec<T> = pattern_fn(test_len).into_iter().map(map_fn).collect();
    check_is_sorted::<T, S>(test_data.as_mut_slice());
}

trait DynTrait: Debug {
    fn get_val(&self) -> i32;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DynValA {
    value: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DynValB {
    value: u64,
}

impl DynTrait for DynValA {
    fn get_val(&self) -> i32 {
        self.value
    }
}
impl DynTrait for DynValB {
    fn get_val(&self) -> i32 {
        let bytes = self.value.to_ne_bytes();
        i32::from_ne_bytes([bytes[0], bytes[1], bytes[6], bytes[7]])
    }
}

impl PartialOrd for dyn DynTrait {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for dyn DynTrait {
    fn cmp(&self, other: &Self) -> Ordering {
        self.get_val().cmp(&other.get_val())
    }
}

impl PartialEq for dyn DynTrait {
    fn eq(&self, other: &Self) -> bool {
        self.get_val() == other.get_val()
    }
}

impl Eq for dyn DynTrait {}

fn shift_i32_to_u32(val: i32) -> u32 {
    (val as i64 + (i32::MAX as i64 + 1)) as u32
}

fn reverse_shift_i32_to_u32(val: u32) -> i32 {
    (val as i64 - (i32::MAX as i64 + 1)) as i32
}

fn extend_i32_to_u64(val: i32) -> u64 {
    // Extends the value into the 64 bit range,
    // while preserving input order.
    (shift_i32_to_u32(val) as u64) * i32::MAX as u64
}

fn extend_i32_to_u128(val: i32) -> u128 {
    // Extends the value into the 64 bit range,
    // while preserving input order.
    (shift_i32_to_u32(val) as u128) * i64::MAX as u128
}

fn dyn_trait_from_i32(val: i32) -> Rc<dyn DynTrait> {
    if val % 2 == 0 {
        Rc::new(DynValA { value: val })
    } else {
        Rc::new(DynValB { value: extend_i32_to_u64(val) })
    }
}

fn i32_from_i32(val: i32) -> i32 {
    val
}

fn i32_from_i32_ref(val: &i32) -> i32 {
    *val
}

fn string_from_i32(val: i32) -> String {
    format!("{:010}", shift_i32_to_u32(val))
}

fn i32_from_string(val: &String) -> i32 {
    reverse_shift_i32_to_u32(val.parse::<u32>().unwrap())
}

fn cell_i32_from_i32(val: i32) -> Cell<i32> {
    Cell::new(val)
}

fn i32_from_cell_i32(val: &Cell<i32>) -> i32 {
    val.get()
}

fn calc_comps_required<T, S: Sort>(v: &mut [T], mut cmp_fn: impl FnMut(&T, &T) -> Ordering) -> u32 {
    let mut comp_counter = 0u32;

    <S as Sort>::sort_by(v, |a, b| {
        comp_counter += 1;

        cmp_fn(a, b)
    });

    comp_counter
}

#[derive(PartialEq, Eq, Debug, Clone)]
#[repr(C)]
struct CompCount {
    val: i32,
    comp_count: Cell<u32>,
}

impl CompCount {
    fn new(val: i32) -> Self {
        Self { val, comp_count: Cell::new(0) }
    }
}

/// Generates $base_name_pattern_name_impl functions calling the test_fns for all test_len.
macro_rules! gen_sort_test_fns {
    (
        $base_name:ident,
        $test_fn:expr,
        $test_lengths:expr,
        [$(($pattern_name:ident, $pattern_fn:expr)),* $(,)?] $(,)?
    ) => {
        $(fn ${concat($base_name, _, $pattern_name, _impl)}<S: Sort>() {
            for test_len in $test_lengths {
                $test_fn(*test_len, $pattern_fn);
            }
        })*
    };
}

/// Generates $base_name_pattern_name_impl functions calling the test_fns for all test_len,
/// with a default set of patterns that can be extended by the caller.
macro_rules! gen_sort_test_fns_with_default_patterns {
    (
        $base_name:ident,
        $test_fn:expr,
        $test_lengths:expr,
        [$(($pattern_name:ident, $pattern_fn:expr)),* $(,)?] $(,)?
    ) => {
        gen_sort_test_fns!(
            $base_name,
            $test_fn,
            $test_lengths,
            [
                (random, patterns::random),
                (random_z1, |len| patterns::random_zipf(len, 1.0)),
                (random_d2, |len| patterns::random_uniform(len, 0..2)),
                (random_d20, |len| patterns::random_uniform(len, 0..16)),
                (random_s95, |len| patterns::random_sorted(len, 95.0)),
                (ascending, patterns::ascending),
                (descending, patterns::descending),
                (saw_mixed, |len| patterns::saw_mixed(
                    len,
                    ((len as f64).log2().round()) as usize
                )),
                $(($pattern_name, $pattern_fn),)*
            ]
        );
    };
}

/// Generates $base_name_type_pattern_name_impl functions calling the test_fns for all test_len for
/// three types that cover the core specialization differences in the sort implementations, with a
/// default set of patterns that can be extended by the caller.
macro_rules! gen_sort_test_fns_with_default_patterns_3_ty {
    (
        $base_name:ident,
        $test_fn:ident,
        [$(($pattern_name:ident, $pattern_fn:expr)),* $(,)?] $(,)?
    ) => {
        gen_sort_test_fns_with_default_patterns!(
            ${concat($base_name, _i32)},
            |len, pattern_fn| $test_fn::<i32, S>(len, i32_from_i32, i32_from_i32_ref, pattern_fn),
            &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
            [$(($pattern_name, $pattern_fn),)*],
        );

        gen_sort_test_fns_with_default_patterns!(
            ${concat($base_name, _cell_i32)},
            |len, pattern_fn| $test_fn::<Cell<i32>, S>(len, cell_i32_from_i32, i32_from_cell_i32, pattern_fn),
            &TEST_LENGTHS[..TEST_LENGTHS.len() - 3],
            [$(($pattern_name, $pattern_fn),)*],
        );

        gen_sort_test_fns_with_default_patterns!(
            ${concat($base_name, _string)},
            |len, pattern_fn| $test_fn::<String, S>(len, string_from_i32, i32_from_string, pattern_fn),
            &TEST_LENGTHS[..TEST_LENGTHS.len() - 3],
            [$(($pattern_name, $pattern_fn),)*],
        );
    };
}

// --- TESTS ---

pub fn basic_impl<S: Sort>() {
    check_is_sorted::<i32, S>(&mut []);
    check_is_sorted::<(), S>(&mut []);
    check_is_sorted::<(), S>(&mut [()]);
    check_is_sorted::<(), S>(&mut [(), ()]);
    check_is_sorted::<(), S>(&mut [(), (), ()]);
    check_is_sorted::<i32, S>(&mut []);
    check_is_sorted::<i32, S>(&mut [77]);
    check_is_sorted::<i32, S>(&mut [2, 3]);
    check_is_sorted::<i32, S>(&mut [2, 3, 6]);
    check_is_sorted::<i32, S>(&mut [2, 3, 99, 6]);
    check_is_sorted::<i32, S>(&mut [2, 7709, 400, 90932]);
    check_is_sorted::<i32, S>(&mut [15, -1, 3, -1, -3, -1, 7]);
}

fn fixed_seed_impl<S: Sort>() {
    let fixed_seed_a = patterns::get_or_init_rand_seed();
    let fixed_seed_b = patterns::get_or_init_rand_seed();

    assert_eq!(fixed_seed_a, fixed_seed_b);
}

fn fixed_seed_rand_vec_prefix_impl<S: Sort>() {
    let vec_rand_len_5 = patterns::random(5);
    let vec_rand_len_7 = patterns::random(7);

    assert_eq!(vec_rand_len_5, vec_rand_len_7[..5]);
}

fn int_edge_impl<S: Sort>() {
    // Ensure that the sort can handle integer edge cases.
    check_is_sorted::<i32, S>(&mut [i32::MIN, i32::MAX]);
    check_is_sorted::<i32, S>(&mut [i32::MAX, i32::MIN]);
    check_is_sorted::<i32, S>(&mut [i32::MIN, 3]);
    check_is_sorted::<i32, S>(&mut [i32::MIN, -3]);
    check_is_sorted::<i32, S>(&mut [i32::MIN, -3, i32::MAX]);
    check_is_sorted::<i32, S>(&mut [i32::MIN, -3, i32::MAX, i32::MIN, 5]);
    check_is_sorted::<i32, S>(&mut [i32::MAX, 3, i32::MIN, 5, i32::MIN, -3, 60, 200, 50, 7, 10]);

    check_is_sorted::<u64, S>(&mut [u64::MIN, u64::MAX]);
    check_is_sorted::<u64, S>(&mut [u64::MAX, u64::MIN]);
    check_is_sorted::<u64, S>(&mut [u64::MIN, 3]);
    check_is_sorted::<u64, S>(&mut [u64::MIN, u64::MAX - 3]);
    check_is_sorted::<u64, S>(&mut [u64::MIN, u64::MAX - 3, u64::MAX]);
    check_is_sorted::<u64, S>(&mut [u64::MIN, u64::MAX - 3, u64::MAX, u64::MIN, 5]);
    check_is_sorted::<u64, S>(&mut [
        u64::MAX,
        3,
        u64::MIN,
        5,
        u64::MIN,
        u64::MAX - 3,
        60,
        200,
        50,
        7,
        10,
    ]);

    let mut large = patterns::random(TEST_LENGTHS[TEST_LENGTHS.len() - 2]);
    large.push(i32::MAX);
    large.push(i32::MIN);
    large.push(i32::MAX);
    check_is_sorted::<i32, S>(&mut large);
}

fn sort_vs_sort_by_impl<S: Sort>() {
    // Ensure that sort and sort_by produce the same result.
    let mut input_normal = [800, 3, -801, 5, -801, -3, 60, 200, 50, 7, 10];
    let expected = [-801, -801, -3, 3, 5, 7, 10, 50, 60, 200, 800];

    let mut input_sort_by = input_normal.to_vec();

    <S as Sort>::sort(&mut input_normal);
    <S as Sort>::sort_by(&mut input_sort_by, |a, b| a.cmp(b));

    assert_eq!(input_normal, expected);
    assert_eq!(input_sort_by, expected);
}

gen_sort_test_fns_with_default_patterns!(
    correct_i32,
    |len, pattern_fn| test_is_sorted::<i32, S>(len, |val| val, pattern_fn),
    TEST_LENGTHS,
    [
        (random_d4, |len| patterns::random_uniform(len, 0..4)),
        (random_d8, |len| patterns::random_uniform(len, 0..8)),
        (random_d311, |len| patterns::random_uniform(len, 0..311)),
        (random_d1024, |len| patterns::random_uniform(len, 0..1024)),
        (random_z1_03, |len| patterns::random_zipf(len, 1.03)),
        (random_z2, |len| patterns::random_zipf(len, 2.0)),
        (random_s50, |len| patterns::random_sorted(len, 50.0)),
        (narrow, |len| patterns::random_uniform(
            len,
            0..=(((len as f64).log2().round()) as i32) * 100
        )),
        (all_equal, patterns::all_equal),
        (saw_mixed_range, |len| patterns::saw_mixed_range(len, 20..50)),
        (pipe_organ, patterns::pipe_organ),
    ]
);

gen_sort_test_fns_with_default_patterns!(
    correct_u64,
    |len, pattern_fn| test_is_sorted::<u64, S>(len, extend_i32_to_u64, pattern_fn),
    TEST_LENGTHS,
    []
);

gen_sort_test_fns_with_default_patterns!(
    correct_u128,
    |len, pattern_fn| test_is_sorted::<u128, S>(len, extend_i32_to_u128, pattern_fn),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

gen_sort_test_fns_with_default_patterns!(
    correct_cell_i32,
    |len, pattern_fn| test_is_sorted::<Cell<i32>, S>(len, Cell::new, pattern_fn),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

gen_sort_test_fns_with_default_patterns!(
    correct_string,
    |len, pattern_fn| test_is_sorted::<String, S>(
        len,
        |val| format!("{:010}", shift_i32_to_u32(val)),
        pattern_fn
    ),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

gen_sort_test_fns_with_default_patterns!(
    correct_f128,
    |len, pattern_fn| test_is_sorted::<F128, S>(len, F128::new, pattern_fn),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

gen_sort_test_fns_with_default_patterns!(
    correct_1k,
    |len, pattern_fn| test_is_sorted::<FFIOneKibiByte, S>(len, FFIOneKibiByte::new, pattern_fn),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

// Dyn values are fat pointers, something the implementation might have overlooked.
gen_sort_test_fns_with_default_patterns!(
    correct_dyn_val,
    |len, pattern_fn| test_is_sorted::<Rc<dyn DynTrait>, S>(len, dyn_trait_from_i32, pattern_fn),
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

fn stability_legacy_impl<S: Sort>() {
    // This non pattern variant has proven to catch some bugs the pattern version of this function
    // doesn't catch, so it remains in conjunction with the other one.

    if <S as Sort>::name().contains("unstable") {
        // It would be great to mark the test as skipped, but that isn't possible as of now.
        return;
    }

    let large_range = if cfg!(miri) { 100..110 } else { 3000..3010 };
    let rounds = if cfg!(miri) { 1 } else { 10 };

    let rand_vals = patterns::random_uniform(5_000, 0..=9);
    let mut rand_idx = 0;

    for len in (2..55).chain(large_range) {
        for _ in 0..rounds {
            let mut counts = [0; 10];

            // create a vector like [(6, 1), (5, 1), (6, 2), ...],
            // where the first item of each tuple is random, but
            // the second item represents which occurrence of that
            // number this element is, i.e., the second elements
            // will occur in sorted order.
            let orig: Vec<_> = (0..len)
                .map(|_| {
                    let n = rand_vals[rand_idx];
                    rand_idx += 1;
                    if rand_idx >= rand_vals.len() {
                        rand_idx = 0;
                    }

                    counts[n as usize] += 1;
                    i32_tup_as_u64((n, counts[n as usize]))
                })
                .collect();

            let mut v = orig.clone();
            // Only sort on the first element, so an unstable sort
            // may mix up the counts.
            <S as Sort>::sort_by(&mut v, |a_packed, b_packed| {
                let a = i32_tup_from_u64(*a_packed).0;
                let b = i32_tup_from_u64(*b_packed).0;

                a.cmp(&b)
            });

            // This comparison includes the count (the second item
            // of the tuple), so elements with equal first items
            // will need to be ordered with increasing
            // counts... i.e., exactly asserting that this sort is
            // stable.
            assert!(v.windows(2).all(|w| i32_tup_from_u64(w[0]) <= i32_tup_from_u64(w[1])));
        }
    }

    // For cpp_sorts that only support u64 we can pack the two i32 inside a u64.
    fn i32_tup_as_u64(val: (i32, i32)) -> u64 {
        let a_bytes = val.0.to_le_bytes();
        let b_bytes = val.1.to_le_bytes();

        u64::from_le_bytes([a_bytes, b_bytes].concat().try_into().unwrap())
    }

    fn i32_tup_from_u64(val: u64) -> (i32, i32) {
        let bytes = val.to_le_bytes();

        let a = i32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let b = i32::from_le_bytes(bytes[4..8].try_into().unwrap());

        (a, b)
    }
}

fn stability_with_patterns<T: Ord + Clone, S: Sort>(
    len: usize,
    type_into_fn: impl Fn(i32) -> T,
    _type_from_fn: impl Fn(&T) -> i32,
    pattern_fn: fn(usize) -> Vec<i32>,
) {
    if <S as Sort>::name().contains("unstable") {
        // It would be great to mark the test as skipped, but that isn't possible as of now.
        return;
    }

    let pattern = pattern_fn(len);

    let mut counts = [0i32; 128];

    // create a vector like [(6, 1), (5, 1), (6, 2), ...],
    // where the first item of each tuple is random, but
    // the second item represents which occurrence of that
    // number this element is, i.e., the second elements
    // will occur in sorted order.
    let orig: Vec<_> = pattern
        .iter()
        .map(|val| {
            let n = val.saturating_abs() % counts.len() as i32;
            counts[n as usize] += 1;
            (type_into_fn(n), counts[n as usize])
        })
        .collect();

    let mut v = orig.clone();
    // Only sort on the first element, so an unstable sort
    // may mix up the counts.
    <S as Sort>::sort(&mut v);

    // This comparison includes the count (the second item
    // of the tuple), so elements with equal first items
    // will need to be ordered with increasing
    // counts... i.e., exactly asserting that this sort is
    // stable.
    assert!(v.windows(2).all(|w| w[0] <= w[1]));
}

gen_sort_test_fns_with_default_patterns_3_ty!(stability, stability_with_patterns, []);

fn observable_is_less<S: Sort>(len: usize, pattern_fn: fn(usize) -> Vec<i32>) {
    // This test, tests that every is_less is actually observable. Ie. this can go wrong if a hole
    // is created using temporary memory and, the whole is used as comparison but not copied back.
    //
    // If this is not upheld a custom type + comparison function could yield UB in otherwise safe
    // code. Eg T == Mutex<Option<Box<str>>> which replaces the pointer with none in the comparison
    // function, which would not be observed in the original slice and would lead to a double free.

    let pattern = pattern_fn(len);
    let mut test_input = pattern.into_iter().map(|val| CompCount::new(val)).collect::<Vec<_>>();

    let mut comp_count_global = 0;

    <S as Sort>::sort_by(&mut test_input, |a, b| {
        a.comp_count.replace(a.comp_count.get() + 1);
        b.comp_count.replace(b.comp_count.get() + 1);
        comp_count_global += 1;

        a.val.cmp(&b.val)
    });

    let total_inner: u64 = test_input.iter().map(|c| c.comp_count.get() as u64).sum();

    assert_eq!(total_inner, comp_count_global * 2);
}

gen_sort_test_fns_with_default_patterns!(
    observable_is_less,
    observable_is_less::<S>,
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

fn panic_retain_orig_set<T: Ord + Clone, S: Sort>(
    len: usize,
    type_into_fn: impl Fn(i32) -> T + Copy,
    type_from_fn: impl Fn(&T) -> i32,
    pattern_fn: fn(usize) -> Vec<i32>,
) {
    let mut test_data: Vec<T> = pattern_fn(len).into_iter().map(type_into_fn).collect();

    let sum_before: i64 = test_data.iter().map(|x| type_from_fn(x) as i64).sum();

    // Calculate a specific comparison that should panic.
    // Ensure that it can be any of the possible comparisons and that it always panics.
    let required_comps = calc_comps_required::<T, S>(&mut test_data.clone(), |a, b| a.cmp(b));
    let panic_threshold = patterns::random_uniform(1, 1..=required_comps as i32)[0] as usize - 1;

    let mut comp_counter = 0;

    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        <S as Sort>::sort_by(&mut test_data, |a, b| {
            if comp_counter == panic_threshold {
                // Make the panic dependent on the test len and some random factor. We want to
                // make sure that panicking may also happen when comparing elements a second
                // time.
                panic!();
            }
            comp_counter += 1;

            a.cmp(b)
        });
    }));

    assert!(res.is_err());

    // If the sum before and after don't match, it means the set of elements hasn't remained the
    // same.
    let sum_after: i64 = test_data.iter().map(|x| type_from_fn(x) as i64).sum();
    assert_eq!(sum_before, sum_after);
}

gen_sort_test_fns_with_default_patterns_3_ty!(panic_retain_orig_set, panic_retain_orig_set, []);

fn panic_observable_is_less<S: Sort>(len: usize, pattern_fn: fn(usize) -> Vec<i32>) {
    // This test, tests that every is_less is actually observable. Ie. this can go wrong if a hole
    // is created using temporary memory and, the whole is used as comparison but not copied back.
    // This property must also hold if the user provided comparison panics.
    //
    // If this is not upheld a custom type + comparison function could yield UB in otherwise safe
    // code. Eg T == Mutex<Option<Box<str>>> which replaces the pointer with none in the comparison
    // function, which would not be observed in the original slice and would lead to a double free.

    let mut test_input =
        pattern_fn(len).into_iter().map(|val| CompCount::new(val)).collect::<Vec<_>>();

    let sum_before: i64 = test_input.iter().map(|x| x.val as i64).sum();

    // Calculate a specific comparison that should panic.
    // Ensure that it can be any of the possible comparisons and that it always panics.
    let required_comps =
        calc_comps_required::<CompCount, S>(&mut test_input.clone(), |a, b| a.val.cmp(&b.val));

    let panic_threshold = patterns::random_uniform(1, 1..=required_comps as i32)[0] as u64 - 1;

    let mut comp_count_global = 0;

    let res = panic::catch_unwind(AssertUnwindSafe(|| {
        <S as Sort>::sort_by(&mut test_input, |a, b| {
            if comp_count_global == panic_threshold {
                // Make the panic dependent on the test len and some random factor. We want to
                // make sure that panicking may also happen when comparing elements a second
                // time.
                panic!();
            }

            a.comp_count.replace(a.comp_count.get() + 1);
            b.comp_count.replace(b.comp_count.get() + 1);
            comp_count_global += 1;

            a.val.cmp(&b.val)
        });
    }));

    assert!(res.is_err());

    let total_inner: u64 = test_input.iter().map(|c| c.comp_count.get() as u64).sum();

    assert_eq!(total_inner, comp_count_global * 2);

    // If the sum before and after don't match, it means the set of elements hasn't remained the
    // same.
    let sum_after: i64 = test_input.iter().map(|x| x.val as i64).sum();
    assert_eq!(sum_before, sum_after);
}

gen_sort_test_fns_with_default_patterns!(
    panic_observable_is_less,
    panic_observable_is_less::<S>,
    &TEST_LENGTHS[..TEST_LENGTHS.len() - 2],
    []
);

fn deterministic<T: Ord + Clone + Debug, S: Sort>(
    len: usize,
    type_into_fn: impl Fn(i32) -> T + Copy,
    type_from_fn: impl Fn(&T) -> i32,
    pattern_fn: fn(usize) -> Vec<i32>,
) {
    // A property similar to stability is deterministic output order. If the entire value is used as
    // the comparison key a lack of determinism has no effect. But if only a part of the value is
    // used as comparison key, a lack of determinism can manifest itself in the order of values
    // considered equal by the comparison predicate.
    //
    // This test only tests that results are deterministic across runs, it does not test determinism
    // on different platforms and with different toolchains.

    let mut test_input =
        pattern_fn(len).into_iter().map(|val| type_into_fn(val)).collect::<Vec<_>>();

    let mut test_input_clone = test_input.clone();

    let comparison_fn = |a: &T, b: &T| {
        let a_i32 = type_from_fn(a);
        let b_i32 = type_from_fn(b);

        let a_i32_key_space_reduced = a_i32 % 10_000;
        let b_i32_key_space_reduced = b_i32 % 10_000;

        a_i32_key_space_reduced.cmp(&b_i32_key_space_reduced)
    };

    <S as Sort>::sort_by(&mut test_input, comparison_fn);
    <S as Sort>::sort_by(&mut test_input_clone, comparison_fn);

    assert_eq!(test_input, test_input_clone);
}

gen_sort_test_fns_with_default_patterns_3_ty!(deterministic, deterministic, []);

fn self_cmp<T: Ord + Clone + Debug, S: Sort>(
    len: usize,
    type_into_fn: impl Fn(i32) -> T + Copy,
    _type_from_fn: impl Fn(&T) -> i32,
    pattern_fn: fn(usize) -> Vec<i32>,
) {
    // It's possible for comparisons to run into problems if the values of `a` and `b` passed into
    // the comparison function are the same reference. So this tests that they never are.

    let mut test_input =
        pattern_fn(len).into_iter().map(|val| type_into_fn(val)).collect::<Vec<_>>();

    let comparison_fn = |a: &T, b: &T| {
        assert_ne!(a as *const T as usize, b as *const T as usize);
        a.cmp(b)
    };

    <S as Sort>::sort_by(&mut test_input, comparison_fn);

    // Check that the output is actually sorted and wasn't stopped by the assert.
    for window in test_input.windows(2) {
        assert!(window[0] <= window[1]);
    }
}

gen_sort_test_fns_with_default_patterns_3_ty!(self_cmp, self_cmp, []);

fn violate_ord_retain_orig_set<T: Ord, S: Sort>(
    len: usize,
    type_into_fn: impl Fn(i32) -> T + Copy,
    type_from_fn: impl Fn(&T) -> i32,
    pattern_fn: fn(usize) -> Vec<i32>,
) {
    // A user may implement Ord incorrectly for a type or violate it by calling sort_by with a
    // comparison function that violates Ord with the orderings it returns. Even under such
    // circumstances the input must retain its original set of elements.

    // Ord implies a strict total order see https://en.wikipedia.org/wiki/Total_order.

    // Generating random numbers with miri is quite expensive.
    let random_orderings_len = if cfg!(miri) { 200 } else { 10_000 };

    // Make sure we get a good distribution of random orderings, that are repeatable with the seed.
    // Just using random_uniform with the same len and range will always yield the same value.
    let random_orderings = patterns::random_uniform(random_orderings_len, 0..2);

    let get_random_0_1_or_2 = |random_idx: &mut usize| {
        let ridx = *random_idx;
        *random_idx += 1;
        if ridx + 1 == random_orderings.len() {
            *random_idx = 0;
        }

        random_orderings[ridx] as usize
    };

    let mut random_idx_a = 0;
    let mut random_idx_b = 0;
    let mut random_idx_c = 0;

    let mut last_element_a = -1;
    let mut last_element_b = -1;

    let mut rand_counter_b = 0;
    let mut rand_counter_c = 0;

    let mut streak_counter_a = 0;
    let mut streak_counter_b = 0;

    // Examples, a = 3, b = 5, c = 9.
    // Correct Ord -> 10010 | is_less(a, b) is_less(a, a) is_less(b, a) is_less(a, c) is_less(c, a)
    let mut invalid_ord_comp_functions: Vec<Box<dyn FnMut(&T, &T) -> Ordering>> = vec![
        Box::new(|_a, _b| -> Ordering {
            // random
            // Eg. is_less(3, 5) == true, is_less(3, 5) == false

            let idx = get_random_0_1_or_2(&mut random_idx_a);
            [Ordering::Less, Ordering::Equal, Ordering::Greater][idx]
        }),
        Box::new(|_a, _b| -> Ordering {
            // everything is less -> 11111
            Ordering::Less
        }),
        Box::new(|_a, _b| -> Ordering {
            // everything is equal -> 00000
            Ordering::Equal
        }),
        Box::new(|_a, _b| -> Ordering {
            // everything is greater -> 00000
            // Eg. is_less(3, 5) == false, is_less(5, 3) == false, is_less(3, 3) == false
            Ordering::Greater
        }),
        Box::new(|a, b| -> Ordering {
            // equal means less else greater -> 01000
            if a == b { Ordering::Less } else { Ordering::Greater }
        }),
        Box::new(|a, b| -> Ordering {
            // Transitive breaker. remember last element -> 10001
            let lea = last_element_a;
            let leb = last_element_b;

            let a_as_i32 = type_from_fn(a);
            let b_as_i32 = type_from_fn(b);

            last_element_a = a_as_i32;
            last_element_b = b_as_i32;

            if a_as_i32 == lea && b_as_i32 != leb { b.cmp(a) } else { a.cmp(b) }
        }),
        Box::new(|a, b| -> Ordering {
            // Sampled random 1% of comparisons are reversed.
            rand_counter_b += get_random_0_1_or_2(&mut random_idx_b);
            if rand_counter_b >= 100 {
                rand_counter_b = 0;
                b.cmp(a)
            } else {
                a.cmp(b)
            }
        }),
        Box::new(|a, b| -> Ordering {
            // Sampled random 33% of comparisons are reversed.
            rand_counter_c += get_random_0_1_or_2(&mut random_idx_c);
            if rand_counter_c >= 3 {
                rand_counter_c = 0;
                b.cmp(a)
            } else {
                a.cmp(b)
            }
        }),
        Box::new(|a, b| -> Ordering {
            // STREAK_LEN comparisons yield a.cmp(b) then STREAK_LEN comparisons less. This can
            // discover bugs that neither, random Ord, or just Less or Greater can find. Because it
            // can push a pointer further than expected. Random Ord will average out how far a
            // comparison based pointer travels. Just Less or Greater will be caught by pattern
            // analysis and never enter interesting code.
            const STREAK_LEN: usize = 50;

            streak_counter_a += 1;
            if streak_counter_a <= STREAK_LEN {
                a.cmp(b)
            } else {
                if streak_counter_a == STREAK_LEN * 2 {
                    streak_counter_a = 0;
                }
                Ordering::Less
            }
        }),
        Box::new(|a, b| -> Ordering {
            // See above.
            const STREAK_LEN: usize = 50;

            streak_counter_b += 1;
            if streak_counter_b <= STREAK_LEN {
                a.cmp(b)
            } else {
                if streak_counter_b == STREAK_LEN * 2 {
                    streak_counter_b = 0;
                }
                Ordering::Greater
            }
        }),
    ];

    for comp_func in &mut invalid_ord_comp_functions {
        let mut test_data: Vec<T> = pattern_fn(len).into_iter().map(type_into_fn).collect();
        let sum_before: i64 = test_data.iter().map(|x| type_from_fn(x) as i64).sum();

        // It's ok to panic on Ord violation or to complete.
        // In both cases the original elements must still be present.
        let _ = panic::catch_unwind(AssertUnwindSafe(|| {
            <S as Sort>::sort_by(&mut test_data, &mut *comp_func);
        }));

        // If the sum before and after don't match, it means the set of elements hasn't remained the
        // same.
        let sum_after: i64 = test_data.iter().map(|x| type_from_fn(x) as i64).sum();
        assert_eq!(sum_before, sum_after);

        if cfg!(miri) {
            // This test is prohibitively expensive in miri, so only run one of the comparison
            // functions. This test is not expected to yield direct UB, but rather surface potential
            // UB by showing that the sum is different now.
            break;
        }
    }
}

gen_sort_test_fns_with_default_patterns_3_ty!(
    violate_ord_retain_orig_set,
    violate_ord_retain_orig_set,
    []
);

macro_rules! instantiate_sort_test_inner {
    ($sort_impl:ty, miri_yes, $test_fn_name:ident) => {
        #[test]
        fn $test_fn_name() {
            $crate::sort::tests::$test_fn_name::<$sort_impl>();
        }
    };
    ($sort_impl:ty, miri_no, $test_fn_name:ident) => {
        #[test]
        #[cfg_attr(miri, ignore)]
        fn $test_fn_name() {
            $crate::sort::tests::$test_fn_name::<$sort_impl>();
        }
    };
}

// Using this construct allows us to get warnings for unused test functions.
macro_rules! define_instantiate_sort_tests {
    ($([$miri_use:ident, $test_fn_name:ident]),*,) => {
        $(pub fn $test_fn_name<S: Sort>() {
            ${concat($test_fn_name, _impl)}::<S>();
        })*


        macro_rules! instantiate_sort_tests_gen {
            ($sort_impl:ty) => {
                $(
                    instantiate_sort_test_inner!(
                        $sort_impl,
                        $miri_use,
                        $test_fn_name
                    );
                )*
            }
        }
    };
}

// Some tests are not tested with miri to avoid prohibitively long test times. This leaves coverage
// holes, but the way they are selected should make for relatively small holes. Many properties that
// can lead to UB are tested directly, for example that the original set of elements is retained
// even when a panic occurs or Ord is implemented incorrectly.
define_instantiate_sort_tests!(
    [miri_yes, basic],
    [miri_yes, fixed_seed],
    [miri_yes, fixed_seed_rand_vec_prefix],
    [miri_yes, int_edge],
    [miri_yes, sort_vs_sort_by],
    [miri_yes, correct_i32_random],
    [miri_yes, correct_i32_random_z1],
    [miri_yes, correct_i32_random_d2],
    [miri_yes, correct_i32_random_d20],
    [miri_yes, correct_i32_random_s95],
    [miri_yes, correct_i32_ascending],
    [miri_yes, correct_i32_descending],
    [miri_yes, correct_i32_saw_mixed],
    [miri_no, correct_i32_random_d4],
    [miri_no, correct_i32_random_d8],
    [miri_no, correct_i32_random_d311],
    [miri_no, correct_i32_random_d1024],
    [miri_no, correct_i32_random_z1_03],
    [miri_no, correct_i32_random_z2],
    [miri_no, correct_i32_random_s50],
    [miri_no, correct_i32_narrow],
    [miri_no, correct_i32_all_equal],
    [miri_no, correct_i32_saw_mixed_range],
    [miri_yes, correct_i32_pipe_organ],
    [miri_no, correct_u64_random],
    [miri_yes, correct_u64_random_z1],
    [miri_no, correct_u64_random_d2],
    [miri_no, correct_u64_random_d20],
    [miri_no, correct_u64_random_s95],
    [miri_no, correct_u64_ascending],
    [miri_no, correct_u64_descending],
    [miri_no, correct_u64_saw_mixed],
    [miri_no, correct_u128_random],
    [miri_yes, correct_u128_random_z1],
    [miri_no, correct_u128_random_d2],
    [miri_no, correct_u128_random_d20],
    [miri_no, correct_u128_random_s95],
    [miri_no, correct_u128_ascending],
    [miri_no, correct_u128_descending],
    [miri_no, correct_u128_saw_mixed],
    [miri_no, correct_cell_i32_random],
    [miri_yes, correct_cell_i32_random_z1],
    [miri_no, correct_cell_i32_random_d2],
    [miri_no, correct_cell_i32_random_d20],
    [miri_no, correct_cell_i32_random_s95],
    [miri_no, correct_cell_i32_ascending],
    [miri_no, correct_cell_i32_descending],
    [miri_no, correct_cell_i32_saw_mixed],
    [miri_no, correct_string_random],
    [miri_yes, correct_string_random_z1],
    [miri_no, correct_string_random_d2],
    [miri_no, correct_string_random_d20],
    [miri_no, correct_string_random_s95],
    [miri_no, correct_string_ascending],
    [miri_no, correct_string_descending],
    [miri_no, correct_string_saw_mixed],
    [miri_no, correct_f128_random],
    [miri_yes, correct_f128_random_z1],
    [miri_no, correct_f128_random_d2],
    [miri_no, correct_f128_random_d20],
    [miri_no, correct_f128_random_s95],
    [miri_no, correct_f128_ascending],
    [miri_no, correct_f128_descending],
    [miri_no, correct_f128_saw_mixed],
    [miri_no, correct_1k_random],
    [miri_yes, correct_1k_random_z1],
    [miri_no, correct_1k_random_d2],
    [miri_no, correct_1k_random_d20],
    [miri_no, correct_1k_random_s95],
    [miri_no, correct_1k_ascending],
    [miri_no, correct_1k_descending],
    [miri_no, correct_1k_saw_mixed],
    [miri_no, correct_dyn_val_random],
    [miri_yes, correct_dyn_val_random_z1],
    [miri_no, correct_dyn_val_random_d2],
    [miri_no, correct_dyn_val_random_d20],
    [miri_no, correct_dyn_val_random_s95],
    [miri_no, correct_dyn_val_ascending],
    [miri_no, correct_dyn_val_descending],
    [miri_no, correct_dyn_val_saw_mixed],
    [miri_no, stability_legacy],
    [miri_no, stability_i32_random],
    [miri_yes, stability_i32_random_z1],
    [miri_no, stability_i32_random_d2],
    [miri_no, stability_i32_random_d20],
    [miri_no, stability_i32_random_s95],
    [miri_no, stability_i32_ascending],
    [miri_no, stability_i32_descending],
    [miri_no, stability_i32_saw_mixed],
    [miri_no, stability_cell_i32_random],
    [miri_yes, stability_cell_i32_random_z1],
    [miri_no, stability_cell_i32_random_d2],
    [miri_no, stability_cell_i32_random_d20],
    [miri_no, stability_cell_i32_random_s95],
    [miri_no, stability_cell_i32_ascending],
    [miri_no, stability_cell_i32_descending],
    [miri_no, stability_cell_i32_saw_mixed],
    [miri_no, stability_string_random],
    [miri_yes, stability_string_random_z1],
    [miri_no, stability_string_random_d2],
    [miri_no, stability_string_random_d20],
    [miri_no, stability_string_random_s95],
    [miri_no, stability_string_ascending],
    [miri_no, stability_string_descending],
    [miri_no, stability_string_saw_mixed],
    [miri_no, observable_is_less_random],
    [miri_yes, observable_is_less_random_z1],
    [miri_no, observable_is_less_random_d2],
    [miri_no, observable_is_less_random_d20],
    [miri_no, observable_is_less_random_s95],
    [miri_no, observable_is_less_ascending],
    [miri_no, observable_is_less_descending],
    [miri_no, observable_is_less_saw_mixed],
    [miri_no, panic_retain_orig_set_i32_random],
    [miri_yes, panic_retain_orig_set_i32_random_z1],
    [miri_no, panic_retain_orig_set_i32_random_d2],
    [miri_no, panic_retain_orig_set_i32_random_d20],
    [miri_no, panic_retain_orig_set_i32_random_s95],
    [miri_no, panic_retain_orig_set_i32_ascending],
    [miri_no, panic_retain_orig_set_i32_descending],
    [miri_no, panic_retain_orig_set_i32_saw_mixed],
    [miri_no, panic_retain_orig_set_cell_i32_random],
    [miri_yes, panic_retain_orig_set_cell_i32_random_z1],
    [miri_no, panic_retain_orig_set_cell_i32_random_d2],
    [miri_no, panic_retain_orig_set_cell_i32_random_d20],
    [miri_no, panic_retain_orig_set_cell_i32_random_s95],
    [miri_no, panic_retain_orig_set_cell_i32_ascending],
    [miri_no, panic_retain_orig_set_cell_i32_descending],
    [miri_no, panic_retain_orig_set_cell_i32_saw_mixed],
    [miri_no, panic_retain_orig_set_string_random],
    [miri_yes, panic_retain_orig_set_string_random_z1],
    [miri_no, panic_retain_orig_set_string_random_d2],
    [miri_no, panic_retain_orig_set_string_random_d20],
    [miri_no, panic_retain_orig_set_string_random_s95],
    [miri_no, panic_retain_orig_set_string_ascending],
    [miri_no, panic_retain_orig_set_string_descending],
    [miri_no, panic_retain_orig_set_string_saw_mixed],
    [miri_no, panic_observable_is_less_random],
    [miri_yes, panic_observable_is_less_random_z1],
    [miri_no, panic_observable_is_less_random_d2],
    [miri_no, panic_observable_is_less_random_d20],
    [miri_no, panic_observable_is_less_random_s95],
    [miri_no, panic_observable_is_less_ascending],
    [miri_no, panic_observable_is_less_descending],
    [miri_no, panic_observable_is_less_saw_mixed],
    [miri_no, deterministic_i32_random],
    [miri_yes, deterministic_i32_random_z1],
    [miri_no, deterministic_i32_random_d2],
    [miri_no, deterministic_i32_random_d20],
    [miri_no, deterministic_i32_random_s95],
    [miri_no, deterministic_i32_ascending],
    [miri_no, deterministic_i32_descending],
    [miri_no, deterministic_i32_saw_mixed],
    [miri_no, deterministic_cell_i32_random],
    [miri_yes, deterministic_cell_i32_random_z1],
    [miri_no, deterministic_cell_i32_random_d2],
    [miri_no, deterministic_cell_i32_random_d20],
    [miri_no, deterministic_cell_i32_random_s95],
    [miri_no, deterministic_cell_i32_ascending],
    [miri_no, deterministic_cell_i32_descending],
    [miri_no, deterministic_cell_i32_saw_mixed],
    [miri_no, deterministic_string_random],
    [miri_yes, deterministic_string_random_z1],
    [miri_no, deterministic_string_random_d2],
    [miri_no, deterministic_string_random_d20],
    [miri_no, deterministic_string_random_s95],
    [miri_no, deterministic_string_ascending],
    [miri_no, deterministic_string_descending],
    [miri_no, deterministic_string_saw_mixed],
    [miri_no, self_cmp_i32_random],
    [miri_yes, self_cmp_i32_random_z1],
    [miri_no, self_cmp_i32_random_d2],
    [miri_no, self_cmp_i32_random_d20],
    [miri_no, self_cmp_i32_random_s95],
    [miri_no, self_cmp_i32_ascending],
    [miri_no, self_cmp_i32_descending],
    [miri_no, self_cmp_i32_saw_mixed],
    [miri_no, self_cmp_cell_i32_random],
    [miri_yes, self_cmp_cell_i32_random_z1],
    [miri_no, self_cmp_cell_i32_random_d2],
    [miri_no, self_cmp_cell_i32_random_d20],
    [miri_no, self_cmp_cell_i32_random_s95],
    [miri_no, self_cmp_cell_i32_ascending],
    [miri_no, self_cmp_cell_i32_descending],
    [miri_no, self_cmp_cell_i32_saw_mixed],
    [miri_no, self_cmp_string_random],
    [miri_yes, self_cmp_string_random_z1],
    [miri_no, self_cmp_string_random_d2],
    [miri_no, self_cmp_string_random_d20],
    [miri_no, self_cmp_string_random_s95],
    [miri_no, self_cmp_string_ascending],
    [miri_no, self_cmp_string_descending],
    [miri_no, self_cmp_string_saw_mixed],
    [miri_no, violate_ord_retain_orig_set_i32_random],
    [miri_yes, violate_ord_retain_orig_set_i32_random_z1],
    [miri_no, violate_ord_retain_orig_set_i32_random_d2],
    [miri_no, violate_ord_retain_orig_set_i32_random_d20],
    [miri_no, violate_ord_retain_orig_set_i32_random_s95],
    [miri_no, violate_ord_retain_orig_set_i32_ascending],
    [miri_no, violate_ord_retain_orig_set_i32_descending],
    [miri_no, violate_ord_retain_orig_set_i32_saw_mixed],
    [miri_no, violate_ord_retain_orig_set_cell_i32_random],
    [miri_yes, violate_ord_retain_orig_set_cell_i32_random_z1],
    [miri_no, violate_ord_retain_orig_set_cell_i32_random_d2],
    [miri_no, violate_ord_retain_orig_set_cell_i32_random_d20],
    [miri_no, violate_ord_retain_orig_set_cell_i32_random_s95],
    [miri_no, violate_ord_retain_orig_set_cell_i32_ascending],
    [miri_no, violate_ord_retain_orig_set_cell_i32_descending],
    [miri_no, violate_ord_retain_orig_set_cell_i32_saw_mixed],
    [miri_no, violate_ord_retain_orig_set_string_random],
    [miri_yes, violate_ord_retain_orig_set_string_random_z1],
    [miri_no, violate_ord_retain_orig_set_string_random_d2],
    [miri_no, violate_ord_retain_orig_set_string_random_d20],
    [miri_no, violate_ord_retain_orig_set_string_random_s95],
    [miri_no, violate_ord_retain_orig_set_string_ascending],
    [miri_no, violate_ord_retain_orig_set_string_descending],
    [miri_no, violate_ord_retain_orig_set_string_saw_mixed],
);

macro_rules! instantiate_sort_tests {
    ($sort_impl:ty) => {
        instantiate_sort_tests_gen!($sort_impl);
    };
}

mod unstable {
    struct SortImpl {}

    impl crate::sort::Sort for SortImpl {
        fn name() -> String {
            "rust_std_unstable".into()
        }

        fn sort<T>(v: &mut [T])
        where
            T: Ord,
        {
            v.sort_unstable();
        }

        fn sort_by<T, F>(v: &mut [T], mut compare: F)
        where
            F: FnMut(&T, &T) -> std::cmp::Ordering,
        {
            v.sort_unstable_by(|a, b| compare(a, b));
        }
    }

    instantiate_sort_tests!(SortImpl);
}

mod stable {
    struct SortImpl {}

    impl crate::sort::Sort for SortImpl {
        fn name() -> String {
            "rust_std_stable".into()
        }

        fn sort<T>(v: &mut [T])
        where
            T: Ord,
        {
            v.sort();
        }

        fn sort_by<T, F>(v: &mut [T], mut compare: F)
        where
            F: FnMut(&T, &T) -> std::cmp::Ordering,
        {
            v.sort_by(|a, b| compare(a, b));
        }
    }

    instantiate_sort_tests!(SortImpl);
}
