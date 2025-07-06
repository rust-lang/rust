//@ run-crash
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ error-pattern: unsafe precondition(s) violated: str::get_unchecked_mut requires
//@ revisions: range range_to range_from backwards_range

fn main() {
    unsafe {
        let mut s: String = "💅".chars().collect();
        let mut s: &mut str = &mut s;
        #[cfg(range)]
        s.get_unchecked_mut(4..5);
        #[cfg(range_to)]
        s.get_unchecked_mut(..5);
        #[cfg(range_from)]
        s.get_unchecked_mut(5..);
        #[cfg(backwards_range)]
        s.get_unchecked_mut(1..0);
    }
}
