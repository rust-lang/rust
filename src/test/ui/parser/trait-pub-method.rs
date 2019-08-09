trait Foo {
    pub fn foo();
    //~^ ERROR unnecessary visibility qualifier
}

fn main() {}
