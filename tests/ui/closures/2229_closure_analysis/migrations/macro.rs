//@ run-rustfix

// See https://github.com/rust-lang/rust/issues/87955

#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here


#[derive(Debug)]
struct Foo(i32);
impl Drop for Foo {
    fn drop(&mut self) {
        println!("{:?} dropped", self.0);
    }
}

fn main() {
    let a = (Foo(0), Foo(1));
    let _ = || dbg!(a.0);
    //~^ ERROR: drop order
    //~| NOTE: will only capture `a.0`
    //~| NOTE: for more information, see
    //~| HELP: add a dummy let to cause `a` to be fully captured
}
//~^ NOTE: dropped here
