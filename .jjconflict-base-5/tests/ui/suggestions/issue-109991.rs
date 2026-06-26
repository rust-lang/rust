struct S {
    a: usize,
    b: usize,
}

fn main() {
    let a: usize;
    let b: usize;
    let c: usize;

    (c) = (&123); //~ ERROR mismatched types
    (a, b) = (123, &mut 123); //~ ERROR mismatched types

    let x: String;
    (x,) = (1,); //~ ERROR mismatched types

    let x: i32;
    [x] = [&1]; //~ ERROR mismatched types

    let x: &i32;
    [x] = [1]; //~ ERROR mismatched types

    let x = (1, &mut 2);
    (a, b) = x; //~ ERROR mismatched types

    S { a, b } = S { a: 1, b: &mut 2 }; //~ ERROR mismatched types
}
