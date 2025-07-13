pub struct Struct;

impl Struct {
    pub fn function(funs: Vec<dyn Fn() -> ()>) {}
    //~^ ERROR the size for values of type
}

struct Vec<T> {
    t: T,
}

fn main() {}

// https://github.com/rust-lang/rust/issues/23281
