trait Foo {
    pub fn foo();
    //~^ ERROR visibility qualifiers are not permitted here
}

fn main() {}
