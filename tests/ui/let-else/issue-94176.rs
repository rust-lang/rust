// Issue #94176: wrong span for the error message of a mismatched type error,
// if the function uses a `let else` construct.


pub fn test(a: Option<u32>) -> Option<u32> { //~ ERROR mismatched types
    let Some(_) = a else { return None; };
    println!("Foo");
}

fn main() {}
