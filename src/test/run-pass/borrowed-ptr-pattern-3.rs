fn foo(s: &r/uint) -> bool {
    match s {
        &3 => true,
        _ => false
    }
}

fn main() {
    assert foo(&3);
    assert !foo(&4);
}
