// error-pattern:mismatched kinds

resource r(i: int) {
}

fn main() {
    // This can't make sense as it would copy the resources
    let i <- [r(0)];
    let j <- [r(1)];
    let k = i + j;
}