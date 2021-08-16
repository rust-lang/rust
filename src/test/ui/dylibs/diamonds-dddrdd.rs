// build-fail

// This fails to compile because the middle static library (m) is included via
// two dylibs: `s_upper` and `t_upper`.

// aux-build: a_basement_dynamic.rs
// aux-build: i_ground_dynamic.rs
// aux-build: j_ground_dynamic.rs
// aux-build: m_middle_rlib.rs
// aux-build: s_upper_dynamic.rs
// aux-build: t_upper_dynamic.rs

extern crate s_upper as s;
extern crate t_upper as t;

fn main() {
    s::s(); t::t();
}
