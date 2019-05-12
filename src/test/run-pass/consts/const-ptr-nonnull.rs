// run-pass

use std::ptr::NonNull;

const DANGLING: NonNull<u32> = NonNull::dangling();
const CASTED: NonNull<u32> = NonNull::cast(NonNull::<i32>::dangling());

fn ident<T>(ident: T) -> T {
    ident
}

pub fn main() {
    assert_eq!(DANGLING, ident(NonNull::dangling()));
    assert_eq!(CASTED, ident(NonNull::dangling()));
}
