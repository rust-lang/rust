// Issue #62007: assigning over a const-index projection of an array
// (in this case, `list[I] = n;`) should in theory be able to kill all borrows
// of `list[0]`, so that `list[0]` could be borrowed on the next
// iteration through the loop.
//
// Currently the compiler does not allow this. We may want to consider
// loosening that restriction in the future. (However, doing so would
// at *least* require T-lang team approval, and probably an RFC; e.g.
// such loosening might make complicate the user's mental mode; it
// also would make code more brittle in the face of refactorings that
// replace constants with variables.

#![allow(dead_code)]

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: [&mut List<T>; 2]) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list[0].value); //~ ERROR cannot borrow `list[_].value` as mutable
        if let Some(n) = list[0].next.as_mut() { //~ ERROR cannot borrow `list[_].next` as mutable
            list[0] = n;
        } else {
            return result;
        }
    }
}

fn main() {}
