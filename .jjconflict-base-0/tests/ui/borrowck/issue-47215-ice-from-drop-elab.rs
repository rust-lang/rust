// rust-lang/rust#47215: at one time, the compiler categorized
// thread-local statics as a temporary rvalue, as a way to enforce
// that they are only valid for a given lifetime.
//
// The problem with this is that you cannot move out of static items,
// but you *can* move temporary rvalues. I.e., the categorization
// above only solves half of the problem presented by thread-local
// statics.

#![feature(thread_local)]

#[thread_local]
static mut X: ::std::sync::atomic::AtomicUsize = ::std::sync::atomic::AtomicUsize::new(0);

fn main() {
    unsafe {
        let mut x = X; //~ ERROR cannot move out of static item `X` [E0507]
        let _y = x.get_mut();
    }
}
