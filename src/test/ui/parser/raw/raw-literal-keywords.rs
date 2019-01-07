fn test_if() {
    r#if true { } //~ ERROR found `true`
                  //~| ERROR cannot find value `if` in this scope
}

fn test_struct() {
    r#struct Test; //~ ERROR found `Test`
                   //~| ERROR cannot find value `struct` in this scope
}

fn test_union() {
    r#union Test; //~ ERROR found `Test`
                  //~| ERROR cannot find value `union` in this scope
}

fn main() {}
