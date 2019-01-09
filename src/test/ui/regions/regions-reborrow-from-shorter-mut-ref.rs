// Issue #8624. Tests that reborrowing the contents of an `&'b mut`
// pointer which is backed by another `&'a mut` can only be done
// for `'a` (which must be a sublifetime of `'b`).

fn copy_borrowed_ptr<'a, 'b>(p: &'a mut &'b mut isize) -> &'b mut isize {
    &mut **p //~ ERROR lifetime mismatch [E0623]
}

fn main() {
    let mut x = 1;
    let mut y = &mut x;
    let z = copy_borrowed_ptr(&mut y);
    *y += 1;
    *z += 1;
}
