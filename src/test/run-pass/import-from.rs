import spam::{ham, eggs};

mod spam {
    fn ham() {}
    fn eggs() {}
}

fn main() {
    ham();
    eggs();
}