//@ run-rustfix

fn foo() -> i32? { //~ ERROR invalid `?` in type
    let x: i32? = Some(1); //~ ERROR invalid `?` in type
    x
}

fn main() {
    let _: Option<i32> = foo();
}
