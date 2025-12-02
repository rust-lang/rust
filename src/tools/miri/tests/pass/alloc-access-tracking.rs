//@compile-flags: -Zmiri-track-alloc-accesses

#[path = "../utils/mod.rs"]
mod utils;

fn main() {
    unsafe {
        let mut b = Box::<[u8; 123]>::new_uninit();
        let ptr = b.as_mut_ptr() as *mut u8;
        utils::miri_track_alloc(ptr);
        *ptr = 42; // Crucially, only a write is printed here, no read!
        assert_eq!(*ptr, 42);
    }
}
