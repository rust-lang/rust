// When checking whether these two impls overlap, we could detect that we
// would require the hidden type of `TAIT` to be equal to both `u32` and `i32`
// and therefore accept them as disjoint. That is annoying to implement with
// the current system because we would have to add the following to each
// returning branch in coherence.
//
//    let _ = infcx.take_opaque_types();
//
// @lcnr: Because of this I decided to not bother and cause this to fail instead.
// In the future we can definitely modify the compiler to accept this
// again.
#![feature(type_alias_impl_trait)]

trait Trait {}

type TAIT = impl Sized;

impl Trait for (TAIT, TAIT) {}

impl Trait for (u32, i32) {}
//~^ ERROR conflicting implementations of trait `Trait` for type `(TAIT, TAIT)`

fn define() -> TAIT {}

fn main() {}
