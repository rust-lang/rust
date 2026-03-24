//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

//! Test that drop_in_place retags the entire place,
//! invalidating all aliases to it.

// A zero-sized drop type -- the retagging of `fn drop` itself won't
// do anything (since it is zero-sized); we are entirely relying on the retagging
// in `drop_in_place` here.
#[repr(transparent)]
struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {
        unsafe {
            // The error really has to mention a protector to make sure we're checking the right thing!
            P.write(0);
            //~[stack]^ ERROR: /not granting access .* because that would remove .* which is strongly protected/
            //~[tree]| ERROR: forbidden
            // For Tree Borrows, the protector is only mentioned in the "help:" texts unfortunately.
        }
    }
}

static mut P: *mut u8 = core::ptr::null_mut();

fn main() {
    unsafe {
        let mut x = (HasDrop, 0u8);
        let x = core::ptr::addr_of_mut!(x);
        P = x.cast();
        core::ptr::drop_in_place(x);
    }
}
