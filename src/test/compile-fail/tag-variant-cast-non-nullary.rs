//error-pattern: non-scalar cast

enum non_nullary {
    nullary,
    other(int),
}

fn main() {
    let v = nullary;
    let val = v as int;
}
