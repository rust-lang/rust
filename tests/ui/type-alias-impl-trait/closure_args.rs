// check-pass

// regression test for https://github.com/rust-lang/rust/issues/100800

#![feature(type_alias_impl_trait)]

trait Anything {}
impl<T> Anything for T {}
type Input = impl Anything;
fn run<F: FnOnce(Input) -> ()>(f: F, i: Input) {
    f(i);
}

fn bop(_: Input) {
    run(
        |x: u32| {
            println!("{x}");
        },
        0,
    );
}

fn main() {}
