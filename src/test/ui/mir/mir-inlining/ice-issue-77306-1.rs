// Regression test for various issues related to normalization & inlining.
// * #68347, #77306, #77668 - missed normalization during inlining.
// * #78442 - missed normalization in validator after inlining.
//
// build-pass
// compile-flags:-Zmir-opt-level=2

pub fn write() {
    create()()
}

pub fn write_generic<T>(_t: T) {
    hide()();
}

pub fn create() -> impl FnOnce() {
   || ()
}

pub fn hide() -> impl Fn() {
    write
}

fn main() {
    write();
    write_generic(());
}
