// run-pass

// All the dependencies are dylibs, so we can successfully link them all and run them.

// aux-build: a_basement_dynamic.rs
// aux-build: i_ground_dynamic.rs
// aux-build: j_ground_dynamic.rs
// aux-build: m_middle_dynamic.rs
// aux-build: s_upper_dynamic.rs
// aux-build: t_upper_dynamic.rs
// aux-build: z_roof_dynamic.rs

extern crate z_roof as z;

mod diamonds_core;

fn main() {
    diamonds_core::sanity_check();
    diamonds_core::check_linked_function_equivalence();
}
