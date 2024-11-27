//@ check-pass
//@ compile-flags: -Znext-solver

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

struct Place {
    field: <&'static [u8] as Mirror>::Assoc,
}

fn main() {
    let local = Place { field: &[] };
    let z = || {
        let y = &local.field[0];
    };
}
