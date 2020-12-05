const N: isize = 1;

enum Foo {
    A = 1,
    B = 1,
    //~^ ERROR discriminant value `1` already exists
    C = 0,
    D,
    //~^ ERROR discriminant value `1` already exists

    E = N,
    //~^ ERROR discriminant value `1` already exists

}

fn main() {}
