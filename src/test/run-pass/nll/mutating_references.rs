// run-pass

struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: &mut List<T>) -> Vec<&mut T> {
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

fn main() {
    let mut list = List { value: 1, next: None };
    let vec = to_refs(&mut list);
    assert_eq!(vec![&mut 1], vec);
}
