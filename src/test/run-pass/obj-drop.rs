

fn main() {
    obj handle(@int i) { }
    // This just tests whether the obj leaks its box state members.

    auto ob = handle(@0xf00f00);
}