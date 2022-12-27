// #60115

mod foo {
    pub bar();
    //~^ ERROR missing `struct` for struct definition
}

fn main() {}
