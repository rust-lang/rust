// build-fail

// This fails to compile because the static library foundations (a, i, j, m) are
// included via two dylibs: `s_upper` and `t_upper`.

// aux-build: a_basement_rlib.rs
// aux-build: i_ground_rlib.rs
// aux-build: j_ground_rlib.rs
// aux-build: m_middle_rlib.rs
// aux-build: s_upper_dynamic.rs
// aux-build: t_upper_dynamic.rs

extern crate s_upper as s;
extern crate t_upper as t;

fn main() {
    s::s(); t::t();
}
