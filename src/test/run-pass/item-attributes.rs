// xfail-stage0

#[attr = "val"]
const int x = 10;

#[attr = "val"]
mod mod1 {
}

#[attr = "val"]
native "rust" mod rustrt {
}

#[attr = "val"]
type t = obj { };

#[attr = "val"]
obj o() { }

fn main() {
}