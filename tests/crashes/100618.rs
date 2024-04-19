//@ known-bug: #100618
//@ compile-flags: -Cdebuginfo=2

//@ only-x86_64
enum Foo<T: 'static> {
    Value(T),
    Recursive(&'static Foo<Option<T>>),
}

fn main() {
    let _x = Foo::Value(());
}
