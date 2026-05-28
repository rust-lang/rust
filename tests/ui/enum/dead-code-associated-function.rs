//@ check-pass
#![warn(dead_code)]

enum E {
    F(),
    C(),
}

impl E {
    #[expect(non_snake_case)]
    fn F() {}
    //~^ WARN: associated items `F` and `C` are never used

    const C: () = ();
}

fn main() {
    let _: E = E::F();
    let _: E = E::C();
}
