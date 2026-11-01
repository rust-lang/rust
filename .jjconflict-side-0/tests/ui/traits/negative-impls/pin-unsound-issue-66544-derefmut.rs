// Demonstrate that "rogue" `DerefMut` impls for `&T` are not allowed.
//
// https://github.com/rust-lang/rust/issues/66544

use std::cell::Cell;
use std::marker::PhantomPinned;
use std::ops::DerefMut;
use std::pin::Pin;

struct MyType<'a>(Cell<Option<&'a mut MyType<'a>>>, PhantomPinned);

impl<'a> DerefMut for &'a MyType<'a> {
    //~^ ERROR E0751
    fn deref_mut(&mut self) -> &mut MyType<'a> {
        self.0.take().unwrap()
    }
}

fn main() {
    let mut unpinned = MyType(Cell::new(None), PhantomPinned);
    let bad_addr = &unpinned as *const MyType<'_> as usize;
    let p = Box::pin(MyType(Cell::new(Some(&mut unpinned)), PhantomPinned));

    // p_ref is okay: it does not point to the bad_addr
    let mut p_ref: Pin<&MyType<'_>> = p.as_ref();
    assert_ne!(bad_addr, &*p_ref as *const _ as usize);

    // but p_mut does point to bad_addr! this is unsound
    let p_mut: Pin<&mut MyType<'_>> = p_ref.as_mut();
    assert_eq!(bad_addr, &*p_mut as *const _ as usize);

    println!("oh no!");
}
