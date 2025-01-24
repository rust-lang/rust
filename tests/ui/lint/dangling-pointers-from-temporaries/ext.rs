#![deny(dangling_pointers_from_temporaries)]

use std::fmt::Debug;

trait Ext1 {
    fn dbg(self) -> Self
    where
        Self: Sized + Debug,
    {
        dbg!(&self);
        self
    }
}

impl<T> Ext1 for *const T {}

trait Ext2 {
    fn foo(self);
}

impl Ext2 for *const u32 {
    fn foo(self) {
        dbg!(unsafe { self.read() });
    }
}

fn main() {
    let _ptr1 = Vec::<u32>::new().as_ptr().dbg();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u32>` will be dropped
    let _ptr2 = vec![0].as_ptr().foo();
    //~^ ERROR a dangling pointer will be produced because the temporary `Vec<u32>` will be dropped
}
