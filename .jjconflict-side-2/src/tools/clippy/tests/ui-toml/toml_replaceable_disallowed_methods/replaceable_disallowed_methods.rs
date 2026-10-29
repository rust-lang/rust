fn bad() {}
fn questionable() {}
fn good() {}

fn main() {
    bad();
    //~^ disallowed_methods
    questionable();
    //~^ disallowed_methods
}
