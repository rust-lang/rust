#![allow(static_mut_refs)]

struct Bar;

static mut DROP_COUNT: usize = 0;

impl Drop for Bar {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1;
        }
    }
}

fn main() {
    let b = [Bar, Bar, Bar, Bar];
    assert_eq!(unsafe { DROP_COUNT }, 0);
    drop(b);
    assert_eq!(unsafe { DROP_COUNT }, 4);

    // check empty case
    let b: [Bar; 0] = [];
    drop(b);
    assert_eq!(unsafe { DROP_COUNT }, 4);
}
