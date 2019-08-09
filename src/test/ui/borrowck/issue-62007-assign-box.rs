// run-pass

// Issue #62007: assigning over a deref projection of a box (in this
// case, `*list = n;`) should be able to kill all borrows of `*list`,
// so that `*list` can be borrowed on the next iteration through the
// loop.

#![allow(dead_code)]

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: Box<&mut List<T>>) -> Vec<&mut T> {
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

fn main() {}
