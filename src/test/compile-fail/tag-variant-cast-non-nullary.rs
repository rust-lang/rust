//error-pattern: non-scalar cast
// black and white have the same discriminator value ...

tag non_nullary {
    nullary;
    other(int);
}

fn main() {
    let v = nullary;
    let val = v as int;
}
