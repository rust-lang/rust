//! This mixes fragments from different pointers to the same allocarion, in a way
//! that we should not accept. See <https://github.com/rust-lang/rust/issues/146291>.
static A: u8 = 123;

const HALF_PTR: usize = std::mem::size_of::<*const ()>() / 2;

const fn mix_ptr() -> *const u8 {
    unsafe {
        let x: *const u8 = &raw const A;
        let mut y = x.wrapping_add(usize::MAX / 4);
        core::ptr::copy_nonoverlapping(
            (&raw const x).cast::<u8>(),
            (&raw mut y).cast::<u8>(),
            HALF_PTR,
        );
        y
    }
}

const APTR: *const u8 = mix_ptr(); //~ERROR: unable to read parts of a pointer

fn main() {
    let a = APTR;
    println!("{a:p}");
    let b = mix_ptr();
    println!("{b:p}");
    assert_eq!(a, b);
}
