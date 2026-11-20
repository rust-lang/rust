enum P {
    C(PC),
}

enum PC {
    Q,
    QA,
}

fn test(proto: P) {
    match proto { //~ ERROR non-exhaustive patterns
        P::C(PC::Q) => (),
    }
}

fn main() {}
