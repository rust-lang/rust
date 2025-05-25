//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

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
