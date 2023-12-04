#![feature(stmt_expr_attributes)]

fn foo() -> String {
    #[cfg(feature = "validation")]
    [1, 2, 3].iter().map(|c| c.to_string()).collect::<String>() //~ ERROR expected `;`, found `#`
    #[cfg(not(feature = "validation"))]
    String::new()
}

fn bar() -> String {
    #[attr]
    [1, 2, 3].iter().map(|c| c.to_string()).collect::<String>() //~ ERROR expected `;`, found `#`
    #[attr] //~ ERROR cannot find attribute `attr` in this scope
    String::new()
}

fn baz() -> String {
    // Issue #118575: Don't ICE when encountering malformed attributes
    #[cfg(feature = "validation")]
    "foo".into()
    #[]
    //~^ ERROR expected identifier, found `]`
    //~| ERROR expected identifier, found `]`
    "bar".into()
}

fn qux() -> String {
    // Issue #118575: Don't ICE when encountering malformed tail expressions
    #[cfg(feature = "validation")]
    "foo".into()
    #[cfg(not(feature = "validation"))] //~ ERROR expected statement after outer attribute
} //~ ERROR expected expression, found `}`

fn main() {
    println!("{}", foo());
}
