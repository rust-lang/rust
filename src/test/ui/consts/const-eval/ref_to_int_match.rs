#![feature(const_fn_union)]

fn main() {
    let n: Int = 40;
    match n {
        0..=10 => {},
        10..=BAR => {}, //~ ERROR lower range bound must be less than or equal to upper
        _ => {},
    }
}

union Foo {
    f: Int,
    r: &'static u32,
}

#[cfg(target_pointer_width="64")]
type Int = u64;

#[cfg(target_pointer_width="32")]
type Int = u32;

const BAR: Int = unsafe { Foo { r: &42 }.f };
