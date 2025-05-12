#[derive(Debug)]
struct Foo {
    #[cfg(all())]
    field: fn(($),), //~ ERROR expected pattern, found `$`
    //~^ ERROR expected pattern, found `$`
}

fn main() {}
