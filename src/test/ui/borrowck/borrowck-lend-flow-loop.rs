#![feature(box_syntax)]

fn borrow(_v: &isize) {}
fn borrow_mut(_v: &mut isize) {}
fn cond() -> bool { panic!() }
fn produce<T>() -> T { panic!(); }

fn inc(v: &mut Box<isize>) {
    *v = box (**v + 1);
}

fn loop_overarching_alias_mut() {
    // In this instance, the borrow ends on the line before the loop

    let mut v: Box<_> = box 3;
    let mut x = &mut v;
    **x += 1;
    loop {
        borrow(&*v); // OK
    }
}

fn block_overarching_alias_mut() {
    // In this instance, the borrow encompasses the entire closure call.

    let mut v: Box<_> = box 3;
    let mut x = &mut v;
    for _ in 0..3 {
        borrow(&*v); //~ ERROR cannot borrow
    }
    *x = box 5;
}
fn loop_aliased_mut() {
    // In this instance, the borrow ends right after each assignment to _x

    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut _x = &w;
    loop {
        borrow_mut(&mut *v); // OK
        _x = &v;
    }
}

fn while_aliased_mut() {
    // In this instance, the borrow ends right after each assignment to _x

    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut _x = &w;
    while cond() {
        borrow_mut(&mut *v); // OK
        _x = &v;
    }
}


fn loop_aliased_mut_break() {
    // In this instance, the borrow ends right after each assignment to _x

    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut _x = &w;
    loop {
        borrow_mut(&mut *v);
        _x = &v;
        break;
    }
    borrow_mut(&mut *v); // OK
}

fn while_aliased_mut_break() {
    // In this instance, the borrow ends right after each assignment to _x

    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut _x = &w;
    while cond() {
        borrow_mut(&mut *v);
        _x = &v;
        break;
    }
    borrow_mut(&mut *v); // OK
}

fn while_aliased_mut_cond(cond: bool, cond2: bool) {
    let mut v: Box<_> = box 3;
    let mut w: Box<_> = box 4;
    let mut x = &mut w;
    while cond {
        **x += 1;
        borrow(&*v); //~ ERROR cannot borrow
        if cond2 {
            x = &mut v; // OK
        }
    }
}
fn loop_break_pops_scopes<'r, F>(_v: &'r mut [usize], mut f: F) where
    F: FnMut(&'r mut usize) -> bool,
{
    // Here we check that when you break out of an inner loop, the
    // borrows that go out of scope as you exit the inner loop are
    // removed from the bitset.

    while cond() {
        while cond() {
            // this borrow is limited to the scope of `r`...
            let r: &'r mut usize = produce();
            if !f(&mut *r) {
                break; // ...so it is not live as exit the `while` loop here
            }
        }
    }
}

fn loop_loop_pops_scopes<'r, F>(_v: &'r mut [usize], mut f: F)
    where F: FnMut(&'r mut usize) -> bool
{
    // Similar to `loop_break_pops_scopes` but for the `loop` keyword

    while cond() {
        while cond() {
            // this borrow is limited to the scope of `r`...
            let r: &'r mut usize = produce();
            if !f(&mut *r) {
                continue; // ...so it is not live as exit (and re-enter) the `while` loop here
            }
        }
    }
}

fn main() {}
