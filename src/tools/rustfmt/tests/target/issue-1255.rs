// Test for issue #1255
// Default annotation incorrectly removed on associated types
#![feature(specialization)]

trait Trait {
    type Type;
}
impl<T> Trait for T {
    default type Type = u64; // 'default' should not be removed
}
