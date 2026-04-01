#![deny(
    clippy::unnecessary_safety_doc,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

/// This is a private function, skip to match behavior with `missing_safety_doc`.
///
/// # Safety
///
/// Boo!
fn you_dont_see_me() {
    //~^ ERROR: safe function's docs have unnecessary `# Safety` section
    unimplemented!();
}

mod private_mod {
    /// This is public but unexported function.
    ///
    /// # Safety
    ///
    /// Very safe!
    pub fn only_crate_wide_accessible() -> Result<(), ()> {
        //~^ ERROR: safe function's docs have unnecessary `# Safety` section
        //~| ERROR: docs for function returning `Result` missing `# Errors` section
        unimplemented!();
    }
}

pub struct S;

impl S {
    /// Private, fine again to stay consistent with `missing_safety_doc`.
    ///
    /// # Safety
    ///
    /// Unnecessary!
    fn private(&self) {
        //~^ ERROR: safe function's docs have unnecessary `# Safety` section
        //~| ERROR: docs for function which may panic missing `# Panics` section
        panic!();
    }
}

#[doc(hidden)]
pub mod __macro {
    pub struct T;
    impl T {
        pub unsafe fn f() {}
        //~^ ERROR: unsafe function's docs are missing a `# Safety` section
    }
}

#[warn(clippy::missing_errors_doc)]
#[test]
fn test() -> Result<(), ()> {
    Ok(())
}

fn main() {}
