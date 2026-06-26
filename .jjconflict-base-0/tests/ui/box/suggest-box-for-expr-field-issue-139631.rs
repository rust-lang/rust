struct X {
    a: Box<u32>,
}

struct Y {
    y: Box<u32>,
}

fn main() {
    let a = 8;
    let v2 = X { a }; //~ ERROR mismatched types [E0308]
    let v3 = Y { y: a }; //~ ERROR mismatched types [E0308]
    let v4 = Y { a }; //~ ERROR struct `Y` has no field named `a` [E0560]
}
