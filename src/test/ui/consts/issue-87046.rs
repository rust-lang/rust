// Regression test for the ICE described in #87046.

#![crate_type="lib"]
#![allow(unreachable_patterns)]

#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub struct Username(str);

pub const ROOT_USER: &Username = Username::from_str("root");

impl Username {
    pub const fn from_str(raw: &str) -> &Self {
        union Transmute<'a> {
            raw: &'a str,
            typed: &'a Username,
        }

        unsafe { Transmute { raw }.typed }
    }

    pub const fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_root(&self) -> bool {
        match self {
            ROOT_USER => true,
            //~^ ERROR: cannot use unsized non-slice type `Username` in constant patterns
            _ => false,
        }
    }
}
