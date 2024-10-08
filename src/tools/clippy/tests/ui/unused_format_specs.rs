#![warn(clippy::unused_format_specs)]
#![allow(unused)]

macro_rules! format_args_from_macro {
    () => {
        format_args!("from macro")
    };
}

fn main() {
    // prints `.`, not `     .`
    println!("{:5}.", format_args!(""));
    //~^ ERROR: format specifiers have no effect on `format_args!()`
    //~| NOTE: `-D clippy::unused-format-specs` implied by `-D warnings`
    //prints `abcde`, not `abc`
    println!("{:.3}", format_args!("abcde"));
    //~^ ERROR: format specifiers have no effect on `format_args!()`

    println!("{:5}.", format_args_from_macro!());
    //~^ ERROR: format specifiers have no effect on `format_args!()`

    let args = format_args!("");
    println!("{args:5}");
    //~^ ERROR: format specifiers have no effect on `format_args!()`
}

fn should_not_lint() {
    println!("{}", format_args!(""));
    // Technically the same as `{}`, but the `format_args` docs specifically mention that you can use
    // debug formatting so allow it
    println!("{:?}", format_args!(""));

    let args = format_args!("");
    println!("{args}");
}
