//@ check-pass
// Regression test for https://github.com/rust-lang/rust/issues/143349

fn main() {
    let x = panic!();
    x.clone();
    //~^ WARN [trait_method_on_coerced_never_type]
    //~| WARN previously accepted
}
