struct Struct<S, T> {
    a: S,
    b: T,
}

fn main() {
    let (mut a, b);
    let mut c;
    let d = Struct { a: 0, b: 1 };
    Struct { a, b, c } = Struct { a: 0, b: 1 }; //~ ERROR does not have a field named `c`
    Struct { a, _ } = Struct { a: 1, b: 2 }; //~ ERROR pattern does not mention field `b`
    //~| ERROR expected identifier, found reserved identifier `_`
    Struct { a, ..d } = Struct { a: 1, b: 2 };
    //~^ ERROR functional record updates are not allowed in destructuring assignments
    Struct { a, .. }; //~ ERROR base expression required after `..`
}
