enum non_nullary {
    nullary,
    other(isize),
}

fn main() {
    let v = non_nullary::nullary;
    let val = v as isize; //~ ERROR non-primitive cast: `non_nullary` as `isize` [E0605]
}
