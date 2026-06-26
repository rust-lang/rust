struct Foo {
    val: (),
}

fn foo() -> Foo { //~ ERROR struct literal body without path
    val: (),
}

fn main() {
    let x = foo();
    x.val == 42; //~ ERROR mismatched types
    let x = { //~ ERROR struct literal body without path
        val: (),
    };
}
