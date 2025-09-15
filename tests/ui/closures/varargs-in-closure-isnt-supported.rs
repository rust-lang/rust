// var-args are not supported in closures, ensure we don't misdirect people (#146489)
#![feature(c_variadic)]

unsafe extern "C" fn thats_not_a_pattern(mut ap: ...) -> u32 {
    let mut lol = |...| (); //~ ERROR: unexpected `...`
    unsafe { ap.arg::<u32>() } //~^ NOTE: C-variadic type `...` is not allowed here
    //~| NOTE: not a valid pattern
}

fn main() {}
