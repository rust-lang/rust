

fn main() {
    obj handle(i: @int) { }
    // This just tests whether the obj leaks its box state members.

    let ob = handle(@0xf00f00);
}