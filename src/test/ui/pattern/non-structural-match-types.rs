// edition:2021
#![allow(incomplete_features)]
#![allow(unreachable_code)]
#![feature(const_async_blocks)]
#![feature(inline_const_pat)]

fn main() {
    match loop {} {
        const { || {} } => {}, //~ ERROR cannot be used in patterns
    }
    match loop {} {
        const { async {} } => {}, //~ ERROR cannot be used in patterns
    }
}
