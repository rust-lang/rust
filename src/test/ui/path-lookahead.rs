// run-pass
// run-rustfix

#![allow(dead_code)]
#![warn(unused_parens)]

// Parser test for #37765

fn with_parens<T: ToString>(arg: T) -> String {
    return (<T as ToString>::to_string(&arg)); //~WARN unnecessary parentheses around `return` value
}

fn no_parens<T: ToString>(arg: T) -> String {
    return <T as ToString>::to_string(&arg);
}

fn main() {}
