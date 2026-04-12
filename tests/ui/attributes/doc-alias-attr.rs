#[doc(alias = "bar")]
fn foo() {}

#[doc(alias("sum", "plus"))]
fn net() {}

struct S;

impl S {
    #[doc(alias("bar"))]
    fn foo() {}

    #[doc(alias= "baz")]
    fn qux(&self, x: i32) {}
}


fn main() {
    S::bar();
    //~^ ERROR no associated function or constant named `bar` found for struct `S` in the current scope
    //~| HELP there is an associated function `foo` with a similar name

    let s = S;
    s.baz(10);
    //~^ ERROR no method named `baz`
    //~| HELP there is a method `qux` with a similar name

    sum(); //~ ERROR: cannot find function `sum` in this scope
           //~| HELP: `net` has a name defined in the doc alias attribute as `sum`

    bar(); //~ ERROR: cannot find function `bar` in this scope
           //~| HELP: `foo` has a name defined in the doc alias attribute as `bar`
}
