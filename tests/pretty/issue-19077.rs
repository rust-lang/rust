//
// Testing that unsafe blocks in match arms are followed by a comma
//@ pp-exact
fn main() {
    match true {
        true if true => (),
        false if false => unsafe {},
        true => {}
        false => (),
    }
}
