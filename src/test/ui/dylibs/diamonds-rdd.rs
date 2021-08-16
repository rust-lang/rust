// build-fail

// This fails to compile because the static library foundation (a) is
// included via two dylibs: `i_ground` and `j_ground`.

// aux-build: a_basement_rlib.rs
// aux-build: i_ground_dynamic.rs
// aux-build: j_ground_dynamic.rs

extern crate i_ground as i;
extern crate j_ground as j;

fn main() {
    i::i(); j::j();
}
