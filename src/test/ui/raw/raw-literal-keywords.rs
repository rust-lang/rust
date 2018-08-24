// compile-flags: -Z parse-only

fn test_if() {
    r#if true { } //~ ERROR found `true`
}

fn test_struct() {
    r#struct Test; //~ ERROR found `Test`
}

fn test_union() {
    r#union Test; //~ ERROR found `Test`
}
