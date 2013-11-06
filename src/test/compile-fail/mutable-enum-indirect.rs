// Tests that an `&` pointer to something inherently mutable is itself
// to be considered mutable.

#[no_freeze]
enum Foo { A }

fn bar<T: Freeze>(_: T) {}

fn main() {
    let x = A;
    bar(&x); //~ ERROR type parameter with an incompatible type
}
