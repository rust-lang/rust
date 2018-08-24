// Test that spans get only base in eager type resolution (structurally_resolve_type).

fn main() {
    let mut x = Default::default();
    x.0;
    //~^ ERROR type annotations needed
    x = 1;
}

fn foo() {
    let mut x = Default::default();
    x[0];
    //~^ ERROR type annotations needed
    x = 1;
}
