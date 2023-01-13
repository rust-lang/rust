mod foo {
    pub fn r#let() {}
    pub fn break() {} //~ ERROR expected identifier, found keyword `break`
}

fn main() {
    foo::let(); //~ ERROR expected identifier, found keyword `let`
    r#break(); //~ ERROR cannot find function `r#break` in this scope
}
