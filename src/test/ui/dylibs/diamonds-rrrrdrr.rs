// run-pass

// You might expect this to fail to compile because the static library
// foundations (a, i, j, m) are included via both a dylib `s_upper` and an rlib
// `t_upper` that are themselves linked via an rlib `z_roof`. But, inexplicably,
// it currently passes.

// aux-build: a_basement_rlib.rs
// aux-build: i_ground_rlib.rs
// aux-build: j_ground_rlib.rs
// aux-build: m_middle_rlib.rs
// aux-build: s_upper_dynamic.rs
// aux-build: t_upper_rlib.rs
// aux-build: z_roof_rlib.rs

extern crate z_roof as z;

mod diamonds_core;

fn main() {
    diamonds_core::sanity_check();
    diamonds_core::check_linked_function_equivalence();
}
