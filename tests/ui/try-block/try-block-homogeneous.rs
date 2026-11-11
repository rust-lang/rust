//@ check-pass
//@ edition: 2018

#![feature(try_blocks)]
#![crate_type = "lib"]

// A few examples of code that only works with homogeneous `try`

pub fn index_or_zero(x: &[i32], i: usize, j: usize) -> i32 {
    // With heterogeneous `try` this fails because
    // it tries to call a method on a type variable.
    try { x.get(i)? + x.get(j)? }.unwrap_or(0)
}

pub fn do_nothing_on_errors(a: &str, b: &str) {
    // With heterogeneous `try` this fails because
    // an underscore pattern doesn't constrain the output type.
    let _ = try {
        let a = a.parse::<i32>()?;
        let b = b.parse::<u32>()?;
        println!("{a} {b}");
    };
}

pub fn print_error_once(a: &str, b: &str) {
    match try {
        let a = a.parse::<i32>()?;
        let b = b.parse::<u32>()?;
        (a, b)
    } {
        Ok(_pair) => {}
        // With heterogeneous `try` this fails because
        // nothing constrains the error type in `e`.
        Err(e) => eprintln!("{e}"),
    }
}
