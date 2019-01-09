use std::cell::RefCell;

pub fn main() {
    let name = RefCell::new("rust");
    let what = RefCell::new("rocks");
    let msg = format!("{name} {}", &*what.borrow(), name=&*name.borrow());
    assert_eq!(msg, "rust rocks".to_string());
}
