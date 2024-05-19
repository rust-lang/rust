// ICE size_and_align_of::<[closure@test.rs:15:5: 17:7]> not supported #88212
// issue: rust-lang/rust#88212
#![feature(unsized_locals)]
//~^ WARN the feature `unsized_locals` is incomplete and may not be safe to use and/or cause compiler crashes

trait Example {}
struct Foo();

impl Example for Foo {}

fn example() -> Box<dyn Example> {
    Box::new(Foo())
}

fn main() {
    let x: dyn Example = *example();
    (move || {
        let _y = x;
        //~^ ERROR the size for values of type `dyn Example` cannot be known at compilation time
    })();
}
