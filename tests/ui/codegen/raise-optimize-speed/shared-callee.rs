//! Two functions marked `#[optimize(speed)]` in different modules share a marked `#[inline]`
//! callee that unmarked code calls too. The callee's shared definition must survive the merging
//! of the units that hold the marked functions, one of which holds a private copy of it.

//@ run-pass
//@ revisions: modules design
//@ compile-flags: -Copt-level=0
//@[modules] compile-flags: -Zraise-optimize-speed=modules,copy-callees
//@[design] compile-flags: -Zraise-optimize-speed=modules,copy-callees,codegen-level,mir

mod a {
    #[optimize(speed)]
    #[inline]
    pub fn shared(v: u32) -> u32 {
        v.wrapping_mul(3)
    }

    // Makes this module's unit of marked functions the larger of the two.
    #[optimize(speed)]
    pub fn big(v: &mut [u32; 64]) {
        for i in 0..64 {
            v[i] = v[i].wrapping_mul(i as u32).rotate_left(3) ^ 0x9e37;
        }
        for i in 1..64 {
            v[i] = v[i].wrapping_add(v[i - 1]);
        }
    }
}

mod b {
    #[optimize(speed)]
    pub fn marked(v: u32) -> u32 {
        crate::a::shared(v) + 1
    }
}

#[inline(never)]
fn unmarked(v: u32) -> u32 {
    a::shared(v)
}

fn main() {
    let mut t = [1u32; 64];
    a::big(&mut t);
    assert_eq!(b::marked(1), 4);
    assert_eq!(unmarked(2), 6);
    assert_ne!(t[63], 1);
}
