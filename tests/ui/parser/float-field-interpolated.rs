struct S(u8, (u8, u8));

macro_rules! generate_field_accesses {
    ($a:tt, $b:literal, $c:expr) => {
        let s = S(0, (0, 0));

        s.$a; // OK
        { s.$b; } //~ ERROR unexpected token: `1.1`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found literal `1.1`
        { s.$c; } //~ ERROR unexpected token: `1.1`
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found expression `1.1`
    };
}

fn main() {
    generate_field_accesses!(1.1, 1.1, 1.1);
}
