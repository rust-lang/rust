struct A {
    foo : i32,
    car : i32,
    barr : i32
}

fn main() {
    let a = A {
        foo : 5,
        bar : 42,
        //~^ ERROR struct `A` has no field named `bar`
    };
}
