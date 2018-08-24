// ignore-tidy-linelength

struct SingleFoo {
    x: i32
}

struct PluralFoo {
    x: i32,
    y: i32,
    z: i32
}

struct TruncatedFoo {
    a: i32,
    b: i32,
    x: i32,
    y: i32,
    z: i32
}

struct TruncatedPluralFoo {
    a: i32,
    b: i32,
    c: i32,
    x: i32,
    y: i32,
    z: i32
}


fn main() {
    let w = SingleFoo { };
    //~^ ERROR missing field `x` in initializer of `SingleFoo`
    let x = PluralFoo {x: 1};
    //~^ ERROR missing fields `y`, `z` in initializer of `PluralFoo`
    let y = TruncatedFoo{x:1};
    //~^ missing fields `a`, `b`, `y` and 1 other field in initializer of `TruncatedFoo`
    let z = TruncatedPluralFoo{x:1};
    //~^ ERROR missing fields `a`, `b`, `c` and 2 other fields in initializer of `TruncatedPluralFoo`
}
