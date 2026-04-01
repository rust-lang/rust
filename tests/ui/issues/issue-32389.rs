//@ run-pass
fn foo<T>() -> T { loop {} }

fn test() {
    let ref mut a: &mut dyn FnMut((i8,), i16) = foo();
    a((0,), 0);
}

fn main() {
    let _ = test;
}
