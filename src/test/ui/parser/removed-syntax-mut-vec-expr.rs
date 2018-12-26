// compile-flags: -Z parse-only

fn f() {
    let v = [mut 1, 2, 3, 4]; //~ ERROR expected expression, found keyword `mut`
}
