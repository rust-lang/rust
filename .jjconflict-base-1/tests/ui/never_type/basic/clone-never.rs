//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/143349

fn main() {
    let x = panic!();
    x.clone();
    //~^ WARN [method_call_on_diverging_infer_var]
    //~| WARN previously accepted
}
