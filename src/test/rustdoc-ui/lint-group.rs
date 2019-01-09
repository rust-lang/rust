//! Documenting the kinds of lints emitted by rustdoc.
//!
//! ```
//! println!("sup");
//! ```

#![deny(rustdoc)]

/// what up, let's make an [error]
///
/// ```
/// println!("sup");
/// ```
pub fn link_error() {} //~^^^^^ ERROR cannot be resolved, ignoring it

/// wait, this doesn't have a doctest?
pub fn no_doctest() {} //~^ ERROR Missing code example in this documentation

/// wait, this *does* have a doctest?
///
/// ```
/// println!("sup");
/// ```
fn private_doctest() {} //~^^^^^ ERROR Documentation test in private item
