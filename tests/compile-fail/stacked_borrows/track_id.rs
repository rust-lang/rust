// compile-flags: -Zmiri-track-id=1372
// do not run on anything but x86_64 linux, because minor libstd changes can change the borrow stack ids
// only-x86_64
// only-linux

use std::mem;

fn main() {
    let mut target = 42;
    // Make sure we cannot use a raw-tagged `&mut` pointing to a frozen location.
    // Even just creating it unfreezes.
    let raw = &mut target as *mut _; // let this leak to raw
    let reference = unsafe { &*raw }; // freeze
    let ptr = reference as *const _ as *mut i32; // raw ptr, with raw tag
    let _mut_ref: &mut i32 = unsafe { mem::transmute(ptr) }; // &mut, with raw tag
    //~^ ERROR disabling item [SharedReadOnly for <1372>] for tag Some(<untagged>)
    // Now we retag, making our ref top-of-stack -- and, in particular, unfreezing.
    let _val = *reference;
}
