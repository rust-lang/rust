struct Foo(bool);
fn foo(_: usize) -> Foo { Foo(false) }

fn main() {
    match Foo(true) {
        foo(x) //~ ERROR expected tuple struct/variant, found function `foo`
        => ()
    }
}
