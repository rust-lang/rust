// Regression test for stack overflow when a `macro_rules!` macro shadows
// a builtin attribute and is used recursively within its own expansion.

//@ check-pass

#![allow(unused)]

#[macro_export]
macro_rules! test {
    () => {
        #[test]
        fn generated_test() {
            assert!(true);
        }
    };
}

// Additional case with derive
#[macro_export]
macro_rules! derive {
    () => {
        #[derive(Debug)]
        struct Foo;
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    test!();
}

mod another_test {
    use super::*;
    derive!();
}

fn main() {}
