// run-pass

// You might expect this to fail to compile because the static library
// foundations (a) is linked via both a dylib `i_ground` and an rlib `j_upper`
// that are themselves linked via an rlib `m_middle`. But, it currently passes,
// presumably due to internal details of rustc's heuristic selection between
// dynamic and static linking.

// aux-build: a_basement_rlib.rs
// aux-build: i_ground_dynamic.rs
// aux-build: j_ground_rlib.rs
// aux-build: m_middle_rlib.rs
// aux-build: s_upper_rlib.rs
// aux-build: t_upper_rlib.rs
// aux-build: z_roof_rlib.rs

extern crate z_roof as z;

mod diamonds_core;

fn main() {
    diamonds_core::sanity_check();
    diamonds_core::check_linked_function_equivalence();
}
