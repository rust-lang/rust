struct Foo {}

impl Foo {
    pub fn bar()
    //~^ ERROR: associated function in `impl` without body
}
//~^ERROR expected one of `->`, `where`, or `{`, found `}`

fn main() {}
