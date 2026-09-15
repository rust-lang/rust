// Ensure const-eval rejects transmute_copy when the destination is larger than the source.
use std::mem::transmute_copy;

const BAD_SIZED: u64 = unsafe { transmute_copy::<u8, u64>(&1u8) };
//~^ ERROR evaluation panicked: unsafe precondition(s) violated
//~| NOTE evaluation of `BAD_SIZED` failed

const THREE: &[u8] = &[1, 2, 3];
const BAD_UNSIZED: u64 = unsafe { transmute_copy::<[u8], u64>(THREE) };
//~^ ERROR evaluation panicked: unsafe precondition(s) violated
//~| NOTE evaluation of `BAD_UNSIZED` failed

trait Foo {}
impl Foo for u8 {}

const FIVE: u8 = 5;
const SMALL_OBJ: &dyn Foo = &FIVE;
const BAD_DYN: u64 = unsafe { transmute_copy::<dyn Foo, u64>(SMALL_OBJ) };
//~^ ERROR evaluation panicked: unsafe precondition(s) violated
//~| NOTE evaluation of `BAD_DYN` failed

fn main() {}
