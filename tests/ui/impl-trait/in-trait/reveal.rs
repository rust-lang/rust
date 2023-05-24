// check-pass

#![feature(return_position_impl_trait_in_trait, refine)]
#![allow(incomplete_features)]

trait Foo {
    fn f() -> Box<impl Sized>;
}

impl Foo for () {
    #[refine]
    fn f() -> Box<String> {
        Box::new(String::new())
    }
}

fn main() {
    let x: Box<String> = <() as Foo>::f();
}
