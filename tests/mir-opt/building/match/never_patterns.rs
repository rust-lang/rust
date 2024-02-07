// skip-filecheck
#![feature(never_patterns)]
#![allow(incomplete_features)]

enum Void {}

// EMIT_MIR never_patterns.opt1.SimplifyCfg-initial.after.mir
fn opt1(res: &Result<u32, Void>) -> &u32 {
    match res {
        Ok(x) => x,
        Err(!),
    }
}

fn main() {
    opt1(&Ok(0));
}
