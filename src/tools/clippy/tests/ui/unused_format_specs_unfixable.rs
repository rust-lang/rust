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
    //prints `abcde`, not `abc`
    println!("{:.3}", format_args!("abcde"));

    println!("{:5}.", format_args_from_macro!());

    let args = format_args!("");
    println!("{args:5}");
}

fn should_not_lint() {
    println!("{}", format_args!(""));
    // Technically the same as `{}`, but the `format_args` docs specifically mention that you can use
    // debug formatting so allow it
    println!("{:?}", format_args!(""));

    let args = format_args!("");
    println!("{args}");
}
