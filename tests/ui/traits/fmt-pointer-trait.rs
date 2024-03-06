//@ run-pass
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    let p: *const u8 = ptr::null();
    let rc = Rc::new(1usize);
    let arc = Arc::new(1usize);
    let b = Box::new("hi");

    let _ = format!("{:p}{:p}{:p}",
                    rc, arc, b);

    if cfg!(target_pointer_width = "32") {
        assert_eq!(format!("{:#p}", p),
                   "0x00000000");
    } else {
        assert_eq!(format!("{:#p}", p),
                   "0x0000000000000000");
    }
    assert_eq!(format!("{:p}", p),
               "0x0");
}
