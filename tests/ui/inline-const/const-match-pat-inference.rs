//@ check-pass

#![feature(inline_const_pat)]

fn main() {
    match 1u64 {
        0 => (),
        const { 0 + 1 } => (),
        const { 2 - 1 } ..= const { u64::MAX } => (),
    }
}
