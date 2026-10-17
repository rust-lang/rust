#[derive(Debug)]
struct Foo {
    #[cfg(true)]
    field: fn(($),), //~ ERROR expected type, found `$`
    //~^ ERROR expected type, found `$`
}

fn main() {}
