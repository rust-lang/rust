// edition:2021
#![allow(incomplete_features)]
#![allow(unreachable_code)]
#![feature(const_async_blocks)]
#![feature(inline_const_pat)]

fn main() {
    match loop {} { //~ ERROR: non-exhaustive patterns: `_` not covered
        const { || {} } => {}, //~ ERROR cannot be used in patterns
    }
    match loop {} { //~ ERROR: non-exhaustive patterns: `_` not covered
        const { async {} } => {}, //~ ERROR cannot be used in patterns
    }
}
