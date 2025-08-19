//@ known-bug: #140123
//@ compile-flags: --crate-type lib

const OK: [&mut [()]; 2] = [empty_mut(), empty_mut()];
const ICE: [&mut [()]; 2] = [const { empty_mut() }; 2];

// Any kind of fn call gets around E0764.
const fn empty_mut() -> &'static mut [()] {
    &mut []
}
