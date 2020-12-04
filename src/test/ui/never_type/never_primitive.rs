// check-pass

// For backwards compatibility, test if a type definition that is
// named `never` shadows the new `never` from the prelude.
#[allow(non_camel_case_types)]
struct never { x: u32 }

fn foo(never: never) -> u32 {
    let never { x } = never;
    x
}

fn main() { }
