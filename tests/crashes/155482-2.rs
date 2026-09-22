//@ known-bug: #155482
//@ only-x86_64
#![feature(generic_assert)]
fn main() {
    std::arch::x86_64::_mm_shuffle_ps(
        todo!(),
        todo!(),
        const {
            assert!(X != 2);
        },
    )
}
