//@ compile-flags: -Znext-solver=globally
//@ check-fail

#![feature(vec_recycle, transmutability)]

fn main() {
    let a: Vec<u8> = vec![0; 100];
    let capacity = a.capacity();
    let addr = a.as_ptr().addr();
    let b: Vec<i8> = a.recycle();
    //~^ ERROR `&MaybeUninit<u8>` cannot be safely transmuted into `&MaybeUninit<i8>` [E0277]
    assert_eq!(b.len(), 0);
    assert_eq!(b.capacity(), capacity);
    assert_eq!(b.as_ptr().addr(), addr);
}
