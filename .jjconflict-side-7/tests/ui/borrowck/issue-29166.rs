//@ run-pass
// This test ensures that vec.into_iter does not overconstrain element lifetime.

pub fn main() {
    original_report();
    revision_1();
    revision_2();
}

fn original_report() {
    drop(vec![&()].into_iter())
}

fn revision_1() {
    // below is what above `vec!` expands into at time of this writing.
    drop(<[_]>::into_vec(::std::boxed::Box::new([&()])).into_iter())
}

fn revision_2() {
    drop((match (Vec::new(), &()) { (mut v, b) => { v.push(b); v } }).into_iter())
}
