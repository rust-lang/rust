#![allow(dead_code)]

// This tests the various kinds of assignments there are. Polonius used to generate `killed`
// facts only on simple assignments, but not projections, incorrectly causing errors to be emitted
// for code accepted by NLL. They are all variations from example code in the NLL RFC.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: polonius_next polonius
//@ check-pass
//@ [polonius_next] compile-flags: -Z polonius=next
//@ [polonius] compile-flags: -Z polonius

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

// Assignment to a local: the `list` assignment should clear the existing
// borrows of `list.value` and `list.next`
fn assignment_to_local<T>(mut list: &mut List<T>) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.value);
        if let Some(n) = list.next.as_mut() {
            list = n;
        } else {
            return result;
        }
    }
}

// Assignment to a deref projection: the `*list` assignment should clear the existing
// borrows of `list.value` and `list.next`
fn assignment_to_deref_projection<T>(mut list: Box<&mut List<T>>) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.value);
        if let Some(n) = list.next.as_mut() {
            *list = n;
        } else {
            return result;
        }
    }
}

// Assignment to a field projection: the `list.0` assignment should clear the existing
// borrows of `list.0.value` and `list.0.next`
fn assignment_to_field_projection<T>(mut list: (&mut List<T>,)) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.0.value);
        if let Some(n) = list.0.next.as_mut() {
            list.0 = n;
        } else {
            return result;
        }
    }
}

// Assignment to a deref field projection: the `*list.0` assignment should clear the existing
// borrows of `list.0.value` and `list.0.next`
fn assignment_to_deref_field_projection<T>(mut list: (Box<&mut List<T>>,)) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.0.value);
        if let Some(n) = list.0.next.as_mut() {
            *list.0 = n;
        } else {
            return result;
        }
    }
}

// Similar to `assignment_to_deref_field_projection` but through a longer projection chain
fn assignment_through_projection_chain<T>(
    mut list: (((((Box<&mut List<T>>,),),),),),
) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut ((((list.0).0).0).0).0.value);
        if let Some(n) = ((((list.0).0).0).0).0.next.as_mut() {
            *((((list.0).0).0).0).0 = n;
        } else {
            return result;
        }
    }
}

fn main() {
}
