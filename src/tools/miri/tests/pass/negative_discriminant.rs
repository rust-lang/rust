enum AB {
    A = -1,
    B = 1,
}

fn main() {
    match AB::A {
        AB::A => (),
        AB::B => panic!(),
    }

    match AB::B {
        AB::A => panic!(),
        AB::B => (),
    }
}
