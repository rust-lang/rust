// check-pass
// known-bug: #91009

#![feature(const_mut_refs)]
#![feature(const_trait_impl)]
struct Panic;
impl const Drop for Panic { fn drop(&mut self) { panic!(); } }
pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &id(&Panic);
};

fn main() {}
