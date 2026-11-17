//! A program with a function marked `#[optimize(speed)]` behaves alike in every form of
//! `-Z raise-optimize-speed`, in which the function is optimized though the crate is not.

//@ run-pass
//@ revisions: off krate unit noipa one_module modules settings design
//@ revisions: added_unused added chosen thinlto
//@ compile-flags: -Copt-level=0
//@[krate] compile-flags: -Zraise-optimize-speed=crate
//@[unit] compile-flags: -Zraise-optimize-speed=unit
//@[noipa] compile-flags: -Zraise-optimize-speed=unit,noipa
//@[one_module] compile-flags: -Zraise-optimize-speed=one-module
//@[modules] compile-flags: -Zraise-optimize-speed=modules
//@[settings] compile-flags: -Zraise-optimize-speed=modules,level=2,codegen-level,mir
//@[design] compile-flags: -Zraise-optimize-speed=modules,copy-callees,codegen-level,mir
//@[added_unused] compile-flags: -Zraise-optimize-speed=modules,probe=added-unused
//@[added] compile-flags: -Zraise-optimize-speed=modules,probe=added
//@[chosen] compile-flags: -Zraise-optimize-speed=modules,probe=chosen
//@[thinlto] compile-flags: -Zraise-optimize-speed=modules -Zthinlto

#[inline(always)]
fn always(x: u32) -> u32 {
    x.wrapping_add(1)
}

fn plain(xs: &[u32]) -> u32 {
    xs.iter().map(|&x| always(x)).fold(0, u32::wrapping_add)
}

#[optimize(speed)]
fn marked(xs: &[u32]) -> u32 {
    xs.iter().map(|&x| always(x)).fold(0, u32::wrapping_add)
}

mod inner {
    #[optimize(speed)]
    pub fn marked(x: u32) -> u32 {
        let double = |x: u32| x.wrapping_mul(2);
        double(x) + super::plain(&[x])
    }
}

fn main() {
    let xs: Vec<u32> = (0..100).collect();
    assert_eq!(plain(&xs), 5050);
    assert_eq!(marked(&xs), 5050);
    assert_eq!(inner::marked(7), 22);
}
