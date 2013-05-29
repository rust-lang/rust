// Tests that if you move from `x.f` or `x[0]`, `x` is inaccessible.
// Also tests that we give a more specific error message.

struct Foo { f: ~str, y: int }
fn consume(_s: ~str) {}
fn touch<A>(_a: &A) {}

fn f10() {
    let x = Foo { f: ~"hi", y: 3 };
    consume(x.f); //~ NOTE `x.f` moved here
    touch(&x.y); //~ ERROR use of partially moved value: `x`
}

fn f20() {
    let x = ~[~"hi"];
    consume(x[0]); //~ NOTE `(*x)[]` moved here
    touch(&x[0]); //~ ERROR use of partially moved value: `x`
}

fn main() {}
