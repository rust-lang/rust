//@aux-build:proc_macros.rs

#![allow(clippy::let_unit_value, clippy::needless_pass_by_ref_mut)]
#![warn(clippy::unnecessary_safety_doc)]

extern crate proc_macros;
use proc_macros::external;

/// This is has no safety section, and does not need one either
pub fn destroy_the_planet() {
    unimplemented!();
}

/// This one does not need a `Safety` section
///
/// # Safety
///
/// This function shouldn't be called unless the horsemen are ready
pub fn apocalypse(universe: &mut ()) {
    //~^ unnecessary_safety_doc
    unimplemented!();
}

/// This is a private function, skip to match behavior with `missing_safety_doc`.
///
/// # Safety
///
/// Boo!
fn you_dont_see_me() {
    unimplemented!();
}

mod private_mod {
    /// This is public but unexported function, skip to match behavior with `missing_safety_doc`.
    ///
    /// # Safety
    ///
    /// Very safe!
    pub fn only_crate_wide_accessible() {
        unimplemented!();
    }

    /// # Safety
    ///
    /// Unnecessary safety!
    pub fn republished() {
        //~^ unnecessary_safety_doc
        unimplemented!();
    }
}

pub use private_mod::republished;

pub trait SafeTraitSafeMethods {
    fn woefully_underdocumented(self);

    /// # Safety
    ///
    /// Unnecessary!
    fn documented(self);
    //~^ unnecessary_safety_doc
}

pub trait SafeTrait {
    fn method();
}

/// # Safety
///
/// Unnecessary!
pub trait DocumentedSafeTrait {
    //~^ unnecessary_safety_doc
    fn method2();
}

pub struct Struct;

impl SafeTraitSafeMethods for Struct {
    fn woefully_underdocumented(self) {
        // all is well
    }

    fn documented(self) {
        // all is still well
    }
}

impl SafeTrait for Struct {
    fn method() {}
}

impl DocumentedSafeTrait for Struct {
    fn method2() {}
}

impl Struct {
    /// # Safety
    ///
    /// Unnecessary!
    pub fn documented() -> Self {
        //~^ unnecessary_safety_doc
        unimplemented!();
    }

    pub fn undocumented(&self) {
        unimplemented!();
    }

    /// Private, fine again to stay consistent with `missing_safety_doc`.
    ///
    /// # Safety
    ///
    /// Unnecessary!
    fn private(&self) {
        unimplemented!();
    }
}

macro_rules! very_safe {
    () => {
        pub fn whee() {
            unimplemented!()
        }

        /// # Safety
        ///
        /// Driving is very safe already!
        pub fn drive() {
            //~^ unnecessary_safety_doc
            whee()
        }
    };
}

very_safe!();

// we don't lint code from external macros
external!(
    pub fn vey_oy() {
        unimplemented!();
    }
);

fn main() {}

// do not lint if any parent has `#[doc(hidden)]` attribute
// see #7347
#[doc(hidden)]
pub mod __macro {
    pub struct T;
    impl T {
        pub unsafe fn f() {}
    }
}

/// # Implementation safety
pub trait DocumentedSafeTraitWithImplementationHeader {
    //~^ unnecessary_safety_doc
    fn method();
}
