use spam::{ham, eggs};

mod spam {
    #[legacy_exports];
    fn ham() { }
    fn eggs() { }
}

fn main() { ham(); eggs(); }
