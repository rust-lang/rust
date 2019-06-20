// run-pass
#![allow(dead_code)]

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: (&mut List<T>,)) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut (list.0).value);
        if let Some(n) = (list.0).next.as_mut() {
            list.0 = n;
        } else {
            return result;
        }
    }
}

fn main() {}
