struct S(u8, (u8, u8));

macro_rules! generate_field_accesses {
    ($a:tt, $b:literal, $c:expr) => {
        let s = S(0, (0, 0));

        s.$a; // OK
        { s.$b; }
        //~^ ERROR unexpected token: ``
        //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found invisible open delimiter
        { s.$c; }
        //~^ ERROR unexpected token: ``
        //~| ERROR expected one of `.`, `;`, `?`, `}`, or an operator, found invisible open delimiter
    };
}

fn main() {
    generate_field_accesses!(1.1, 1.1, 1.1);
}

// njn: error messages aren't good, could do better
