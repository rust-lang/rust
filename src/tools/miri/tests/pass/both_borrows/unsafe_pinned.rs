//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(unsafe_pinned)]

use std::pin::UnsafePinned;

fn mutate(x: &UnsafePinned<i32>) {
    let ptr = x as *const _ as *mut i32;
    unsafe { ptr.write(42) };
}

fn mut_alias(x: &mut UnsafePinned<i32>, y: &mut UnsafePinned<i32>) {
    unsafe {
        x.get().write(0);
        y.get().write(0);
        x.get().write(0);
        y.get().write(0);
    }
}

// Also try this with a type for which we implement `Unpin`, just to be extra mean.
struct MyUnsafePinned<T>(UnsafePinned<T>);
impl<T> Unpin for MyUnsafePinned<T> {}

fn my_mut_alias(x: &mut MyUnsafePinned<i32>, y: &mut MyUnsafePinned<i32>) {
    unsafe {
        x.0.get().write(0);
        y.0.get().write(0);
        x.0.get().write(0);
        y.0.get().write(0);
    }
}

fn main() {
    let mut x = UnsafePinned::new(0i32);
    mutate(&x);
    assert_eq!(unsafe { x.get().read() }, 42);

    let ptr = &raw mut x;
    unsafe { mut_alias(&mut *ptr, &mut *ptr) };

    let ptr = ptr.cast::<MyUnsafePinned<i32>>();
    unsafe { my_mut_alias(&mut *ptr, &mut *ptr) };
}
