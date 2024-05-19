//FIXME(compiler-errors): This fixup should suggest the full box path, not just `Box`

struct Box<T> {
    t: T,
}

struct Foo {
    //~^ ERROR recursive type `Foo` has infinite size
    inner: Foo,
}

fn main() {}
