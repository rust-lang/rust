// check-pass

#![feature(return_position_impl_trait_in_trait, lint_reasons)]
#![allow(incomplete_features)]

pub trait Foo {
    fn f() -> Box<impl Sized>;
}

impl Foo for () {
    #[expect(refining_impl_trait)]
    fn f() -> Box<String> {
        Box::new(String::new())
    }
}

fn main() {
    let x: Box<String> = <() as Foo>::f();
}
