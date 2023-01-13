// run-pass
// ignore-pretty issue #37195

#[path = "mod_dir_simple"]
mod biscuits {
    pub mod test;
}

#[path = "mod_dir_simple"]
mod gravy {
    pub mod test;
}

pub fn main() {
    assert_eq!(biscuits::test::foo(), 10);
    assert_eq!(gravy::test::foo(), 10);
}
