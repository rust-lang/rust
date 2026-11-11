struct Foo(u32, u32);
impl Foo {
    fn foo(&self) {
        match *self {
            Foo::(1, 2) => {}, //~ HELP: consider removing the `::` here to turn this into a tuple struct pattern
            //~^ NOTE: while parsing this parenthesized list of type arguments starting
            //~^^ ERROR: expected type, found `1`
            //~^^^ NOTE: expected type
             _ => {},
        }
    }
}

fn main() {}
