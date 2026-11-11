#![feature(gca_min_const_items, gca_macroless_args)]
#![expect(incomplete_features)]

trait Tr {
    const N: usize;
}

struct Blah<const N: usize>;

fn foo() -> Blah<{ Tr::N }> {
    //~^ ERROR ambiguous associated constant
    todo!()
}

fn main() {}
