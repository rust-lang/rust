// Tests that bindings to move-by-default values trigger moves of the
// discriminant. Also tests that the compiler explains the move in
// terms of the binding, not the discriminant.

struct Foo<A> { f: A }
fn guard(_s: String) -> bool {panic!()}
fn touch<A>(_a: &A) {}

fn f10() {
    let x = Foo {f: "hi".to_string()};

    let y = match x {
        Foo {f} => {}
    };

    touch(&x); //~ ERROR borrow of moved value: `x`
    //~^ value borrowed here after partial move
    //~| move occurs because `x.f` has type `std::string::String`
}

fn main() {}
