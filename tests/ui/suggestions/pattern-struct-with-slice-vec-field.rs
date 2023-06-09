use std::ops::Deref;

struct Foo {
    v: Vec<u32>,
}

struct Bar {
    v: Vec<u32>,
}

impl Deref for Bar {
    type Target = Vec<u32>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

fn f(foo: &Foo) {
    match foo {
        Foo { v: [1, 2] } => {}
        //~^ ERROR expected an array or slice, found `Vec<u32>
        _ => {}
    }
}

fn bar(bar: &Bar) {
    match bar {
        Bar { v: [1, 2] } => {}
        //~^ ERROR expected an array or slice, found `Vec<u32>
        _ => {}
    }
}

fn main() {}
