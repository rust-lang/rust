//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

//@[stack]error-in-other-file: which is strongly protected
//@[tree]error-in-other-file: /deallocation through .* is forbidden/

struct Newtype<'a>(&'a mut i32);

fn dealloc_while_running(_n: Newtype<'_>, dealloc: impl FnOnce()) {
    dealloc();
}

// Make sure that we protect references inside structs.
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
