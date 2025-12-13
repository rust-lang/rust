#[derive(Debug)]
struct Foo {
    #[cfg(true)]
    field: fn(($),), //~ ERROR expected pattern, found `$`
    //~^ ERROR expected pattern, found `$`
}

fn main() {}
