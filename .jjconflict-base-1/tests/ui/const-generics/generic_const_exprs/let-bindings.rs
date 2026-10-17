#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

// We do not yet want to support let-bindings in abstract consts,
// so this test should keep failing for now.
fn test<const N: usize>() -> [u8; { let x = N; N + 1 }] where [u8; { let x = N; N + 1 }]: Default {
    //~^ ERROR overly complex generic constant
    //~| ERROR overly complex generic constant
    Default::default()
}

fn main() {
    let x = test::<31>();
    assert_eq!(x, [0; 32]);
}
