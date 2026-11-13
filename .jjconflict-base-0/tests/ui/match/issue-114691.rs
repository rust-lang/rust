//@ run-pass

// This test used to be miscompiled by LLVM 17.
#![allow(dead_code)]

enum Pass {
    Opaque {
        clear_color: [f32; 4],
        with_depth_pre_pass: bool,
    },
    Transparent,
}

enum LoadOp {
    Clear,
    Load,
}

#[inline(never)]
fn check(x: Option<LoadOp>) {
    assert!(x.is_none());
}

#[inline(never)]
fn test(mode: Pass) {
    check(match mode {
        Pass::Opaque {
            with_depth_pre_pass: true,
            ..
        }
        | Pass::Transparent => None,
        _ => Some(LoadOp::Clear),
    });
}

fn main() {
    println!("Hello, world!");
    test(Pass::Transparent);
}
