// Ensure that taking a mutable raw ptr to an uninitialized variable does not change its
// initializedness.

struct S;

fn main() {
    let mut x: S;
    std::ptr::addr_of_mut!(x); //~ borrow of

    let y = x; // Should error here if `addr_of_mut` is ever allowed on uninitialized variables
    drop(y);
}
