enum E {
    V(u8)
}

impl E {
    fn V() {}
}

fn main() {
    <E>::V(); //~ ERROR this function takes 1 parameter but 0 parameters were supplied
}
