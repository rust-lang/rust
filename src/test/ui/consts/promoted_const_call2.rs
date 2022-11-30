// check-pass
// known-bug: #91009

#![feature(const_precise_live_drops)]
pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &id(&String::new());
};

fn main() {}
