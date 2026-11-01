//@ build-pass
//@ compile-flags: --crate-type lib

pub mod one {
    pub const OK: [&mut [()]; 2] = [empty_mut(), empty_mut()];
    pub const ICE: [&mut [()]; 2] = [const { empty_mut() }; 2];

    // Any kind of fn call gets around E0764.
    const fn empty_mut() -> &'static mut [()] {
        &mut []
    }
}

pub mod three {
    pub const ICE: [&mut [()]; 2] = [const { empty_mut() }; 2];

    const fn empty_mut() -> &'static mut [()] {
        unsafe { std::slice::from_raw_parts_mut(std::ptr::dangling_mut(), 0) }
    }
}

pub mod four {
    const ICE: [&mut [(); 0]; 2] = [const { empty_mut() }; 2];

    const fn empty_mut() -> &'static mut [(); 0] {
        &mut []
    }
    // https://github.com/rust-lang/rust/issues/140123#issuecomment-2820664450
    const ICE2: [&mut [(); 0]; 2] = [const {
        let x = &mut [];
        x
    }; 2];
}
