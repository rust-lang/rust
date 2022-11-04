//! Test that drop_in_place mutably retags the entire place,
//! ensuring it is writeable

//@error-pattern: /retag .* for Unique permission .* only grants SharedReadOnly permission/

#[repr(transparent)]
struct HasDrop;

impl Drop for HasDrop {
    fn drop(&mut self) {}
}

fn main() {
    unsafe {
        let x = (0u8, HasDrop);
        let x = core::ptr::addr_of!(x);
        core::ptr::drop_in_place(x.cast_mut());
    }
}
