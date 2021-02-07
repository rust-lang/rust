// Check that an `enum` variant is resolved, in the value namespace,
// with higher priority than other inherent items when there is a conflict.

enum E {
    V(u8)
}

impl E {
    fn V() {}
}

enum E2 {
    V,
}

impl E2 {
    const V: u8 = 0;
}

fn main() {
    <E>::V(); //~ ERROR arguments to this function are incorrect
    let _: u8 = <E2>::V; //~ ERROR mismatched types
}
