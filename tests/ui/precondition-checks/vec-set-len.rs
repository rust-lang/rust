//@ run-crash
//@ compile-flags: -Cdebug-assertions=yes
//@ error-pattern: Vec::set_len requires that new_len <= capacity()

fn main() {
    let mut vec: Vec<i32> = Vec::with_capacity(5);
    // Test set_len with length > capacity
    unsafe {
        vec.set_len(10);
    }
}
