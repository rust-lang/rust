// Double-check we didn't go too far with our resolution to issue
// #62007: assigning over a field projection (`list.1 = n;` in this
// case) should kill only borrows of `list.1`; `list.0` can *not*
// necessarily be borrowed on the next iteration through the loop.

#![allow(dead_code)]

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<'a, T>(mut list: (&'a mut List<T>, &'a mut List<T>)) -> Vec<&'a mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut (list.0).value); //~ ERROR cannot borrow `list.0.value` as mutable
        if let Some(n) = (list.0).next.as_mut() { //~ ERROR cannot borrow `list.0.next` as mutable
            list.1 = n;
        } else {
            return result;
        }
    }
}

fn main() {}
