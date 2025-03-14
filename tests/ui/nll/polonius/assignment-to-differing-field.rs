#![allow(dead_code)]

// Compared to `assignment-kills-loans.rs`, we check here
// that we do not kill too many borrows. Assignments to the `.1`
// field projections should leave the borrows on `.0` intact.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius legacy
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}


fn assignment_to_field_projection<'a, T>(
    mut list: (&'a mut List<T>, &'a mut List<T>),
) -> Vec<&'a mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut (list.0).value);
        //~^ ERROR cannot borrow `list.0.value` as mutable

        if let Some(n) = (list.0).next.as_mut() {
        //~^ ERROR cannot borrow `list.0.next` as mutable
            list.1 = n;
        } else {
            return result;
        }
    }
}

fn assignment_through_projection_chain<'a, T>(
    mut list: (((((Box<&'a mut List<T>>, Box<&'a mut List<T>>),),),),),
) -> Vec<&'a mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut ((((list.0).0).0).0).0.value);
        //~^ ERROR cannot borrow `list.0.0.0.0.0.value` as mutable

        if let Some(n) = ((((list.0).0).0).0).0.next.as_mut() {
        //~^ ERROR cannot borrow `list.0.0.0.0.0.next` as mutable
            *((((list.0).0).0).0).1 = n;
        } else {
            return result;
        }
    }
}

fn main() {}
