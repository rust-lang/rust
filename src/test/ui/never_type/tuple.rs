#[allow(dead_code)]

fn foo() {
    let mut a = (return, );
    a.0 = 1;
    a.0 = 1.1; //~ ERROR mismatched types
}

fn bar() {
    let mut a = (return, );
    a.0.test(); //~ ERROR type annotations needed for `(_,)`
}

fn baz() {
    let mut a = (return, );
    a + 1; //~ ERROR cannot add `{integer}` to `(_,)`
}

fn main() {}
