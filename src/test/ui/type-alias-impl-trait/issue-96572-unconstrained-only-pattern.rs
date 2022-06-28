#![feature(type_alias_impl_trait)]
// check-pass

type T = impl Copy;

fn foo(foo: T) {
    let (mut x, mut y) = foo;
    x = 42;
    y = "foo";
}

type U = impl Copy;

fn bar(bar: Option<U>) {
    match bar {
        Some((mut x, mut y)) => {
            x = 42;
            y = "foo";
        }
        None => {}
    }
}

fn main() {}
