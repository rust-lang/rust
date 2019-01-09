#![allow(non_snake_case)]

#[derive(Copy, Clone)]
enum Q { R(Option<usize>) }

fn xyzzy(q: Q) -> usize {
    match q {
        Q::R(S) if S.is_some() => { 0 }
        _ => 1
    }
}


pub fn main() {
    assert_eq!(xyzzy(Q::R(Some(5))), 0);
}
