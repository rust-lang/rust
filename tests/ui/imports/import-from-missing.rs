use spam::{ham, eggs}; //~ ERROR unresolved import `spam::eggs` [E0432]
                       //~^ no `eggs` in `spam`

mod spam {
    pub fn ham() { }
}

fn main() {
    ham();
    // Expect eggs to pass because the compiler inserts a fake name for it
    eggs();
}
