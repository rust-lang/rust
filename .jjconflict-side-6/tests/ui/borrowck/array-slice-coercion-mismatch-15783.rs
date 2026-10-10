//! Regression test for https://github.com/rust-lang/rust/issues/15783

//@ dont-require-annotations: NOTE

pub fn foo(params: Option<&[&str]>) -> usize {
    params.unwrap().first().unwrap().len()
}

fn main() {
    let name = "Foo";
    let x = Some(&[name]);
    let msg = foo(x);
    //~^ ERROR mismatched types
    //~| NOTE expected enum `Option<&[&str]>`
    //~| NOTE found enum `Option<&[&str; 1]>`
    //~| NOTE expected `Option<&[&str]>`, found `Option<&[&str; 1]>`
    assert_eq!(msg, 3);
}
