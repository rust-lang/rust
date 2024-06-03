//! This test used to ICE because const blocks didn't have a body
//! anymore, making a lot of logic very fragile around handling the
//! HIR of a const block.
//! https://github.com/rust-lang/rust/issues/125846

//@ check-pass

#![feature(inline_const_pat)]

fn main() {
    match 0 {
        const {
            let a = 10_usize;
            *&a
        }
        | _ => {}
    }
}
