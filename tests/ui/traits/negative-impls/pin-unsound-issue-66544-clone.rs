use std::cell::Cell;
use std::marker::PhantomPinned;
use std::pin::Pin;

struct MyType<'a>(Cell<Option<&'a mut MyType<'a>>>, PhantomPinned);

impl<'a> Clone for &'a mut MyType<'a> {
    //~^ ERROR E0751
    fn clone(&self) -> &'a mut MyType<'a> {
        self.0.take().unwrap()
    }
}

fn main() {
    let mut unpinned = MyType(Cell::new(None), PhantomPinned);
    let bad_addr = &unpinned as *const MyType<'_> as usize;
    let mut p = Box::pin(MyType(Cell::new(Some(&mut unpinned)), PhantomPinned));

    // p_mut1 is okay: it does not point to the bad_addr
    let p_mut1: Pin<&mut MyType<'_>> = p.as_mut();
    assert_ne!(bad_addr, &*p_mut1 as *const _ as usize);

    // but p_mut2 does point to bad_addr! this is unsound
    let p_mut2: Pin<&mut MyType<'_>> = p_mut1.clone();
    assert_eq!(bad_addr, &*p_mut2 as *const _ as usize);
}
