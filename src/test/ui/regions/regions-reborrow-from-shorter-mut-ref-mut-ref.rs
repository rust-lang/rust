// Issue #8624. Test for reborrowing with 3 levels, not just two.

fn copy_borrowed_ptr<'a, 'b, 'c>(p: &'a mut &'b mut &'c mut isize) -> &'b mut isize {
    &mut ***p //~ ERROR 14:5: 14:14: lifetime mismatch [E0623]
}

fn main() {
}
