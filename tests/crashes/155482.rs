//@ known-bug: #155482
//@ only-x86_64
#![feature(generic_assert)]
fn main() {
    const I = 0;
    std::arch::x86_64::_mm_shuffle_ps(todo!(), todo!(), const {
        assert!(N != 0);
    });
}
