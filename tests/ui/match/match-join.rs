// a good test that we merge paths correctly in the presence of a
// variable that's used before it's declared

fn my_panic() -> ! { panic!(); }

fn main() {
    match true { false => { my_panic(); } true => { } }

    println!("{}", x); //~ ERROR cannot find value `x` in this scope
    let x: isize;
}
