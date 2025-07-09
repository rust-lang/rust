//@ run-fail
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=no -Zub-checks=yes
//@ check-run-results
//@ revisions: toolarge badalign

fn main() {
    unsafe {
        #[cfg(toolarge)]
        std::alloc::Layout::from_size_align_unchecked(isize::MAX as usize, 2);
        #[cfg(badalign)]
        std::alloc::Layout::from_size_align_unchecked(1, 3);
    }
}
