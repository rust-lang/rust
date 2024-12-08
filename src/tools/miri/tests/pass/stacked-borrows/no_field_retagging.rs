//@compile-flags: -Zmiri-retag-fields=none

struct Newtype<'a>(#[allow(dead_code)] &'a mut i32);

fn dealloc_while_running(_n: Newtype<'_>, dealloc: impl FnOnce()) {
    dealloc();
}

// Make sure that we do *not* retag the fields of `Newtype`.
fn main() {
    let ptr = Box::into_raw(Box::new(0i32));
    #[rustfmt::skip] // I like my newlines
    unsafe {
        dealloc_while_running(
            Newtype(&mut *ptr),
            || drop(Box::from_raw(ptr)),
        )
    };
}
