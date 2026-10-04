fn f() -> isize { //~ ERROR mismatched types
    // Make sure typestate doesn't interpret this match expression as
    // the function result
   match true { true => { } _ => {} };
}

fn main() { }
