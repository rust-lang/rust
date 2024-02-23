#![feature(never_type)]

fn test2() {
    let x: !;
    let c2 = SingleVariant::Points(0)
        | match x { //~ ERROR no implementation for `SingleVariant | ()`
            _ => (),
        };
}

enum SingleVariant {
    Points(u32),
}

fn main() {}
