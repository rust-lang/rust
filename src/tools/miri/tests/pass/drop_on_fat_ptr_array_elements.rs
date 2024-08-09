#![allow(static_mut_refs)]

trait Foo {}

struct Bar;

impl Foo for Bar {}

static mut DROP_COUNT: usize = 0;

impl Drop for Bar {
    fn drop(&mut self) {
        unsafe {
            DROP_COUNT += 1;
        }
    }
}

fn main() {
    let b: [Box<dyn Foo>; 4] = [Box::new(Bar), Box::new(Bar), Box::new(Bar), Box::new(Bar)];
    assert_eq!(unsafe { DROP_COUNT }, 0);
    drop(b);
    assert_eq!(unsafe { DROP_COUNT }, 4);
}
