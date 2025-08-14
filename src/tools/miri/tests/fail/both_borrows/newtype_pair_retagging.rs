//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

//@[stack]error-in-other-file: which is strongly protected
//@[tree]error-in-other-file: /deallocation through .* is forbidden/
struct Newtype<'a>(#[allow(dead_code)] &'a mut i32, #[allow(dead_code)] i32);

fn dealloc_while_running(_n: Newtype<'_>, dealloc: impl FnOnce()) {
    dealloc();
}

// Make sure that we protect references inside structs that are passed as ScalarPair.
fn main() {
    let ptr = Box::into_raw(Box::new(0i32));
    #[rustfmt::skip] // I like my newlines
    unsafe {
        dealloc_while_running(
            Newtype(&mut *ptr, 0),
            || drop(Box::from_raw(ptr)),
        )
    };
}
