use std::mem;

#[allow(unused)]
struct MySlice {
    prefix: u64,
    tail: [u8],
}

#[cfg(target_pointer_width = "64")]
const TOO_BIG: usize = 1usize << 47;
#[cfg(target_pointer_width = "32")]
const TOO_BIG: usize = 1usize << 31;

fn main() { unsafe {
    let ptr = Box::into_raw(Box::new(0u8));
    // The slice part is actually not "too big", but together with the `prefix` field it is.
    let _x: &MySlice = mem::transmute((ptr, TOO_BIG-1)); //~ ERROR: invalid reference metadata: total size is bigger than largest supported object
} }
