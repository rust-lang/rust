pub struct Thing;
//@no-rustfix
pub fn has_thing(things: &[Thing]) -> bool {
    let is_thing_ready = |_peer: &Thing| -> bool { todo!() };
    things.iter().find(|p| is_thing_ready(p)).is_some()
    //~^ ERROR: called `is_some()` after searching an `Iterator` with `find`
    //~| NOTE: `-D clippy::search-is-some` implied by `-D warnings`
}

fn main() {}
