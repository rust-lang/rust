struct S(u8, (u8, u8));

macro_rules! generate_field_accesses {
    ($a:tt, $b:literal, $c:expr) => {
        let s = S(0, (0, 0));

        s.$a; // OK
        { s.$b; } //~ ERROR unexpected token: `literal` metavariable
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `literal` metavariable
        { s.$c; } //~ ERROR unexpected token: `expr` metavariable
                  //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found `expr` metavariable
    };
}

fn main() {
    generate_field_accesses!(1.1, 1.1, 1.1);
}
