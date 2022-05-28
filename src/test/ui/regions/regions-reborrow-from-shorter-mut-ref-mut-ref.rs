// Issue #8624. Test for reborrowing with 3 levels, not just two.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn copy_borrowed_ptr<'a, 'b, 'c>(p: &'a mut &'b mut &'c mut isize) -> &'b mut isize {
    &mut ***p
    //[base]~^ ERROR lifetime mismatch [E0623]
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
}
