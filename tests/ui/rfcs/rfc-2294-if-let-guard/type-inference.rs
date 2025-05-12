//@ check-pass

#![feature(if_let_guard)]

struct S;

fn get<T>() -> Option<T> {
    None
}

fn main() {
    match get() {
        x if let Some(S) = x => {}
        _ => {}
    }
}
