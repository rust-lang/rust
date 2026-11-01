//@ run-pass

#[path = "mod_dir_simple"]
mod pancakes {
    #[path = "test.rs"]
    pub mod syrup;
}

pub fn main() {
    assert_eq!(pancakes::syrup::foo(), 10);
}
