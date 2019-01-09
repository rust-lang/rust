enum NonNullary {
    Nullary,
    Other(isize),
}

fn main() {
    let v = NonNullary::Nullary;
    let val = v as isize; //~ ERROR non-primitive cast: `NonNullary` as `isize` [E0605]
}
