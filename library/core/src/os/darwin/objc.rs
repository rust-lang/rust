//! Defines types and macros for Objective-C interoperability.

#![unstable(feature = "darwin_objc", issue = "145496")]
#![allow(nonstandard_style)]

use crate::fmt;

/// Equivalent to Objective-C’s `struct objc_class` type.
#[repr(u8)]
pub enum objc_class {
    #[unstable(
        feature = "objc_class_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant1,
    #[unstable(
        feature = "objc_class_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant2,
}

impl fmt::Debug for objc_class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("objc_class").finish()
    }
}

/// Equivalent to Objective-C’s `struct objc_selector` type.
#[repr(u8)]
pub enum objc_selector {
    #[unstable(
        feature = "objc_selector_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant1,
    #[unstable(
        feature = "objc_selector_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant2,
}

impl fmt::Debug for objc_selector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("objc_selector").finish()
    }
}

/// Equivalent to Objective-C’s `Class` type.
pub type Class = *mut objc_class;

/// Equivalent to Objective-C’s `SEL` type.
pub type SEL = *mut objc_selector;

/// Gets a reference to an Objective-C class.
///
/// This macro will yield an expression of type [`Class`] for the given class name string literal.
///
/// # Example
///
/// ```no_run
/// #![feature(darwin_objc)]
/// use core::os::darwin::objc;
///
/// let string_class = objc::class!("NSString");
/// ```
#[allow_internal_unstable(rustc_attrs)]
pub macro class($classname:expr) {{
    // Since static Objective-C class references actually end up with multiple definitions
    // across dylib boundaries, we only expose the value of the static and don't provide a way to
    // get the address of or a reference to the static.
    unsafe extern "C" {
        #[rustc_objc_class = $classname]
        safe static VAL: $crate::os::darwin::objc::Class;
    }
    VAL
}}

/// Gets a reference to an Objective-C selector.
///
/// This macro will yield an expression of type [`SEL`] for the given method name string literal.
///
/// It is similar to Objective-C’s `@selector` directive.
///
/// # Examples
///
/// ```no_run
/// #![feature(darwin_objc)]
/// use core::os::darwin::objc;
///
/// let alloc_sel = objc::selector!("alloc");
/// let init_sel = objc::selector!("initWithCString:encoding:");
/// ```
#[allow_internal_unstable(rustc_attrs)]
pub macro selector($methname:expr) {{
    // Since static Objective-C selector references actually end up with multiple definitions
    // across dylib boundaries, we only expose the value of the static and don't provide a way to
    // get the address of or a reference to the static.
    unsafe extern "C" {
        #[rustc_objc_selector = $methname]
        safe static VAL: $crate::os::darwin::objc::SEL;
    }
    VAL
}}
