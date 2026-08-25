//@ run-pass

#[path = "mod_dir_simple"]
mod pancakes {
    pub mod test;
}

pub fn main() {
    assert_eq!(pancakes::test::foo(), 10);
}
