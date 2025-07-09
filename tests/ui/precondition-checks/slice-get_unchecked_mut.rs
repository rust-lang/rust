//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: slice::get_unchecked_mut requires
//@ revisions: usize range range_to range_from backwards_range

fn main() {
    unsafe {
        let mut s = &mut [0];
        #[cfg(usize)]
        s.get_unchecked_mut(1);
        #[cfg(range)]
        s.get_unchecked_mut(1..2);
        #[cfg(range_to)]
        s.get_unchecked_mut(..2);
        #[cfg(range_from)]
        s.get_unchecked_mut(2..);
        #[cfg(backwards_range)]
        s.get_unchecked_mut(1..0);
     }
}
