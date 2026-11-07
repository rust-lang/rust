fn test_if() {
    r#if true { } //~ ERROR found keyword `true`
}

fn test_struct() {
    r#struct Test; //~ ERROR found `Test`
}

fn test_union() {
    r#union Test; //~ ERROR found `Test`
}

fn test_if_2() {
    let _ = r#if; //~ ERROR cannot find value `r#if` in this scope
}

fn test_struct_2() {
    let _ = r#struct; //~ ERROR cannot find value `r#struct` in this scope
}

fn test_union_2() {
    let _ = r#union; //~ ERROR cannot find value `union` in this scope
}

fn main() {}
