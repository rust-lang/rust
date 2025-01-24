//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: index out of bounds: the len is 2 but the index is 2
//@ revisions: oob_a oob_b

fn main() {
    let mut pair = [0u8; 2];
    unsafe {
        #[cfg(oob_a)]
        pair.swap(0, 2);
        #[cfg(oob_b)]
        pair.swap(2, 0);
    }
}
