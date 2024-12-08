//@ known-bug: #103507

#![feature(const_trait_impl)]

struct Panic;
impl const Drop for Panic { fn drop(&mut self) { panic!(); } }

pub const fn id<T>(x: T) -> T { x }
pub const C: () = {
    let _: &'static _ = &id(&Panic);
    //FIXME ~^ ERROR: temporary value dropped while borrowed
    //FIXME ~| ERROR: temporary value dropped while borrowed
};

fn main() {
    let _: &'static _ = &id(&Panic);
    //FIXME ~^ ERROR: temporary value dropped while borrowed
    //FIXME ~| ERROR: temporary value dropped while borrowed
    let _: &'static _ = &&(Panic, 0).1;
    //FIXME~^ ERROR: temporary value dropped while borrowed
    //FIXME~| ERROR: temporary value dropped while borrowed
}
