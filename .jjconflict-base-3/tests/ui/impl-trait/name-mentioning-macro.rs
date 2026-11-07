trait Foo<T> {}

macro_rules! bar {
    () => { () }
}

fn foo(x: impl Foo<bar!()>) {
    let () = x;
    //~^ ERROR mismatched types
}

fn main() {}
