// check-pass

struct Foo {
    bar: dyn for<'r> Fn(usize, &'r dyn FnMut())
}

fn main() {
}
