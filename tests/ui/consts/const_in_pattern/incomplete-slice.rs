#[derive(PartialEq)]
enum E {
    A,
}

const E_SL: &[E] = &[E::A];

fn main() {
    match &[][..] {
        //~^ ERROR match is non-exhaustive [E0004]
        E_SL => {}
        //~^ WARN to use a constant of type `E` in a pattern, `E` must be annotated with `#[derive(PartialEq, Eq)]`
        //~| WARN this was previously accepted by the compiler but is being phased out
    }
}
