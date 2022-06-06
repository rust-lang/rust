const N: isize = 1;

enum Foo {
    //~^ ERROR discriminant value `1` assigned more than once
    //~| ERROR discriminant value `1` assigned more than once
    //~| ERROR discriminant value `1` assigned more than once
    A = 1,
    B = 1,
    C = 0,
    D,

    E = N,

}

fn main() {}
