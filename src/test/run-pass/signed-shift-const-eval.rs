enum test { thing = -5 >> 1u }
fn main() {
    assert(thing as int == -3);
}
