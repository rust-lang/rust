#![allow(static_mut_refs)]

struct Bar(u16); // ZSTs are tested separately

static mut DROP_COUNT: usize = 0;

impl Drop for Bar {
    fn drop(&mut self) {
        assert_eq!(self.0 as usize, unsafe { DROP_COUNT }); // tests whether we are called at a valid address
        unsafe {
            DROP_COUNT += 1;
        }
    }
}

fn main() {
    let b = [Bar(0), Bar(1), Bar(2), Bar(3)];
    assert_eq!(unsafe { DROP_COUNT }, 0);
    drop(b);
    assert_eq!(unsafe { DROP_COUNT }, 4);

    // check empty case
    let b: [Bar; 0] = [];
    drop(b);
    assert_eq!(unsafe { DROP_COUNT }, 4);
}
