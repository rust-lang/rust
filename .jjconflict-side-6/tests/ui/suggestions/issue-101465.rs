#![feature(trait_alias)]

struct B;
struct C;

trait Tr {}

impl Tr for B {}
impl Tr for C {}

trait Tr2<S> = Into<S>;

fn foo2<T: Tr2<()>>() {}

fn foo() -> impl Tr {
    let x = foo2::<_>();

    match true {
        true => B,
        false => C,
        //~^ ERROR `match` arms have incompatible types
    }
}

fn main() {}
