// compile-flags: -Z parse-only

fn f() {
    let a_box = box mut 42; //~ ERROR expected expression, found keyword `mut`
}
