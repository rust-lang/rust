struct Foo {}

impl Foo {
    pub fn bar()
    //~^ ERROR: expected `;`, found `}`
    //~| ERROR: associated function in `impl` without body
}

fn main() {}
