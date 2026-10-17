#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Never {}

fn example(x: Never) -> [i32; 1] {
    let ! = x;
    [1]
}

fn function_param_never(!: Never) -> [i32; 1] {
    [1]
}

fn generic_never<T>(!: T) -> [i32; 1] //~ ERROR mismatched types
where
    T: Copy,
{
    [1]
}

fn main() {
    let _ = "12".lines().map(|!| [1]); //~ ERROR mismatched types
}
