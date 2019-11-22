#![allow(const_err)]

// error-pattern: referenced constant has errors

fn main() {
    let x = &(1 / (1 - 1));
}
