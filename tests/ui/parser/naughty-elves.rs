struct Foo;

impl Foo {
    fn bar(&smut elf) {}
    //~^ ERROR expected one of `:`, `@`, or `|`
    //~| HELP consider making this elf less naughty
    //~| SUGGESTION &mut self
}

fn main() {}
