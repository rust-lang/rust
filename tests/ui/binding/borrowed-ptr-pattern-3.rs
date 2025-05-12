//@ run-pass

fn foo<'r>(s: &'r usize) -> bool {
    match s {
        &3 => true,
        _ => false
    }
}

pub fn main() {
    assert!(foo(&3));
    assert!(!foo(&4));
}
