// #60115

mod foo {
    pub bar();
    //~^ ERROR missing `fn` or `struct` for function or struct definition
}

fn main() {}
