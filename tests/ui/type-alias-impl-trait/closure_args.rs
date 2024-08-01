//@ check-pass

// regression test for https://github.com/rust-lang/rust/issues/100800

#![feature(type_alias_impl_trait)]

mod foo {
    pub trait Anything {}
    impl<T> Anything for T {}
    pub type Input = impl Anything;

    fn bop(_: Input) {
        super::run(
            |x: u32| {
                println!("{x}");
            },
            0,
        );
    }
}
use foo::Input;

fn run<F: FnOnce(Input) -> ()>(f: F, i: Input) {
    f(i);
}

fn main() {}
