//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: NonZero::from_mut_unchecked requires

#![feature(nonzero_from_mut)]

fn main() {
    unsafe {
        let mut num = 0u8;
        std::num::NonZeroU8::from_mut_unchecked(&mut num);
    }
}
