// run-pass

// There is no sharing of an rlib via two dylibs, and thus we can link and run
// this program.

// aux-build: a_basement_dynamic.rs
// aux-build: i_ground_dynamic.rs
// aux-build: j_ground_dynamic.rs
// aux-build: m_middle_dynamic.rs
// aux-build: s_upper_rlib.rs
// aux-build: t_upper_rlib.rs
// aux-build: z_roof_rlib.rs

extern crate z_roof as z;

mod diamonds_core;

fn main() {
    diamonds_core::sanity_check();
    diamonds_core::check_linked_function_equivalence();
}
