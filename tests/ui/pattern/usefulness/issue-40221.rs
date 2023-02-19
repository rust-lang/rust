enum P {
    C(PC),
}

enum PC {
    Q,
    QA,
}

fn test(proto: P) {
    match proto { //~ ERROR match is non-exhaustive
        P::C(PC::Q) => (),
    }
}

fn main() {}
