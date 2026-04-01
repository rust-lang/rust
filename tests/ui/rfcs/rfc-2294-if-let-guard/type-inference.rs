//@ check-pass

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
