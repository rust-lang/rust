// run-pass
#![allow(dead_code)]

static X2: u64 = !0 as u16 as u64;
static Y2: u64 = !0 as u32 as u64;
const X: u64 = !0 as u16 as u64;
const Y: u64 = !0 as u32 as u64;

fn main() {
    assert_eq!(match 1 {
        X => unreachable!(),
        Y => unreachable!(),
        _ => 1
    }, 1);
}
