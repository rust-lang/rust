// a good test that we merge paths correctly in the presence of a
// variable that's used before it's declared

fn my_fail() -> ! { fail; }

fn main() {
    match true { false => { my_fail(); } true => { } }

    log(debug, x); //~ ERROR unresolved name: x
    let x: int;
}
