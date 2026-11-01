trait Trait {
    fn method() -> impl Sized;
}

// Ensure that we don't try to probe the name of the RPITIT when looking for
// fixes to suggest for the missing associated type's trait path below.

fn foo() -> i32::Assoc {}
//~^ ERROR ambiguous associated type

fn main() {}
