// build-pass (FIXME(62277): could be check-pass?)

struct Foo {
    bar: dyn for<'r> Fn(usize, &'r dyn FnMut())
}

fn main() {
}
