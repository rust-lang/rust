//@ run-pass

fn foo(s: &String) -> bool {
    match &**s {
        "kitty" => true,
        _ => false
    }
}

pub fn main() {
    assert!(foo(&"kitty".to_string()));
    assert!(!foo(&"gata".to_string()));
}
