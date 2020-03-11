// build-fail

fn main() {
    Bug::V([0; !0]); //~ ERROR is too big for the current
}

enum Bug {
    V([u8; !0]),
}
