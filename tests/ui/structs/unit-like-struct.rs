//@ run-pass
struct Foo;

pub fn main() {
    let x: Foo = Foo;
    match x {
        Foo => { println!("hi"); }
    }
}
