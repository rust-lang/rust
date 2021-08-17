// run-pass

// Make use of new `-C prefer-dynamic=...` flag to allow *only* `std` to be linked dynamically.

// no-prefer-dynamic
// compile-flags: -C prefer-dynamic=std -Z prefer-dynamic-std

// aux-build: a_basement_both.rs
// aux-build: i_ground_both.rs
// aux-build: j_ground_both.rs
// aux-build: m_middle_both.rs
// aux-build: s_upper_both.rs
// aux-build: t_upper_both.rs
// aux-build: z_roof_both.rs

extern crate z_roof as z;

mod diamonds_core;

fn main() {
    diamonds_core::sanity_check();
    diamonds_core::check_linked_function_equivalence();
}
