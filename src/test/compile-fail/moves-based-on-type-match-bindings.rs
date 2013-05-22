// Tests that bindings to move-by-default values trigger moves of the
// discriminant. Also tests that the compiler explains the move in
// terms of the binding, not the discriminant.

struct Foo<A> { f: A }
fn guard(_s: ~str) -> bool {fail!()}
fn touch<A>(_a: &A) {}

fn f10() {
    let x = Foo {f: ~"hi"};

    let y = match x {
        Foo {f} => {} //~ NOTE moved here
    };

    touch(&x); //~ ERROR use of partially moved value: `x`
}

fn main() {}
