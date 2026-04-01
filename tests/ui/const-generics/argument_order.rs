//@ reference: items.generics.syntax.decl-order

struct Good<const N: usize, T> {
    arr: [u8; { N }],
    another: T,
}

struct Bad<const N: usize, 'a, T, 'b, const M: usize, U> {
    //~^ ERROR lifetime parameters must be declared prior
    a: &'a T,
    b: &'b U,
}

fn main() {
    let _: Bad<7, 'static, u32, 'static, 17, u16>;
    //~^ ERROR lifetime provided when a type was expected
 }
