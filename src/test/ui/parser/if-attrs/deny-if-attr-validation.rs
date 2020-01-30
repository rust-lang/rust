fn bar() {}
fn foo() {
    bar(#[cfg(FALSE)] if true {}); //~ ERROR attributes are not yet allowed
}


fn main() {

    #[cfg(FALSE)] //~ ERROR attributes are not yet allowed on `if` expressions
    if true { panic!() }

    #[cfg_attr(FALSE, allow(warnings))] //~ ERROR attributes are not yet allowed on `if` expressions
    if true { panic!() }

    #[allow(warnings)] if true {} //~ ERROR attributes are not yet allowed on `if` expressions
    if false {
    } else if true {
    }

    #[allow(warnings)] if let Some(_) = Some(true) { //~ ERROR attributes are not yet allowed
    }
}
