//@ known-bug: #140123
//@ compile-flags: --crate-type lib

const ICE: [&mut [(); 0]; 2] = [const { empty_mut() }; 2];

const fn empty_mut() -> &'static mut [(); 0] {
    &mut []
}
// https://github.com/rust-lang/rust/issues/140123#issuecomment-2820664450
const ICE2: [&mut [(); 0]; 2] = [const {
    let x = &mut [];
    x
}; 2];
