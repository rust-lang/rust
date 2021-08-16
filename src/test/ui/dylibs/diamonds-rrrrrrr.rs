// run-pass

// All the dependencies are rlibs, so we can successfully link them all and run them.

// aux-build: a_basement_rlib.rs
// aux-build: i_ground_rlib.rs
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
