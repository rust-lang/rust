// Test that we detect nested calls that could free pointers evaluated
// for earlier arguments.



fn rewrite(v: &mut Box<usize>) -> usize {
    *v = Box::new(22);
    **v
}

fn add(v: &usize, w: usize) -> usize {
    *v + w
}

fn implicit() {
    let mut a: Box<_> = Box::new(1);

    // Note the danger here:
    //
    //    the pointer for the first argument has already been
    //    evaluated, but it gets freed when evaluating the second
    //    argument!
    add(
        &*a,
        rewrite(&mut a)); //~ ERROR cannot borrow
}

fn explicit() {
    let mut a: Box<_> = Box::new(1);
    add(
        &*a,
        rewrite(&mut a)); //~ ERROR cannot borrow
}

fn main() {}
