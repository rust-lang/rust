struct Foo {
    v: Vec<u32>,
}

fn f(foo: &Foo) {
    match foo {
        Foo { v: [1, 2] } => {}
        //~^ ERROR expected an array or slice, found `Vec<u32>
        _ => {}
    }
}

fn main() {}
