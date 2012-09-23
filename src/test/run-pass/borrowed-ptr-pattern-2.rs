fn foo(s: &~str) -> bool {
    match s {
        &~"kitty" => true,
        _ => false
    }
}

fn main() {
    assert foo(&~"kitty");
    assert !foo(&~"gata");
}
