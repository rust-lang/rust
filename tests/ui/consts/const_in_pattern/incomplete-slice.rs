#[derive(PartialEq)]
enum E {
    A,
}

const E_SL: &[E] = &[E::A];

fn main() {
    match &[][..] {
        //~^ ERROR non-exhaustive patterns: `&[]` and `&[_, _, ..]` not covered [E0004]
        E_SL => {}
    }
}
