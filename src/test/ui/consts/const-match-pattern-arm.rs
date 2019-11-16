#![allow(warnings)]

const x: bool = match Some(true) { //~ ERROR `match` is not allowed in a `const`
    Some(value) => true,
    _ => false
};

const y: bool = {
    match Some(true) { //~ ERROR `match` is not allowed in a `const`
        Some(value) => true,
        _ => false
    }
};

fn main() {}
