//compile-pass

struct Foo {
    bar: for<'r> Fn(usize, &'r FnMut())
}

fn main() {
}

