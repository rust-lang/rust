fn thing(x: impl FnOnce(&u32)) {}

fn main() {
    let f = |_| ();
    thing(f); //~ERROR mismatched types
}
