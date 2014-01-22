// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Primitive traits representing basic 'kinds' of types

Rust types can be classified in various useful ways according to
intrinsic properties of the type. These classifications, often called
'kinds', are represented as traits.

They cannot be implemented by user code, but are instead implemented
by the compiler automatically for the types to which they apply.

*/

/// Types able to be transferred across task boundaries.
#[lang="send"]
pub trait Send {
    // empty.
}

/// Types that are either immutable or have inherited mutability.
#[lang="freeze"]
pub trait Freeze {
    // empty.
}

/// Types with a constant size known at compile-time.
#[lang="sized"]
pub trait Sized {
    // Empty.
}

/// Types that can be copied by simply copying bits (i.e. `memcpy`).
///
/// The name "POD" stands for "Plain Old Data" and is borrowed from C++.
#[lang="pod"]
pub trait Pod {
    // Empty.
}

/// A 0-sized struct that is specially recognized as not a POD.
///
/// Structs and enums containing this value will also not be POD
///
/// # Example
///
/// ```rust
/// use std::kinds::NotPod;
///
/// struct Foo {
///     np: NotPod
/// }
///
/// fn test<T: Pod>(t: T) {}
///
/// // This will fail to compile becuase `Foo` is not freezeable
/// # { fn test<T>() {}
/// test(Foo { np: NotPod  })
/// # }
/// ```
#[lang="not_pod"]
#[cfg(not(stage0))]
#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
pub struct NotPod;

/// A 0-sized struct that is specially recognized as not `Freeze`.
///
/// Structs and enums containing this value will also not be `Freeze`
///
/// # Example
///
/// ```rust
/// use std::kinds::NotFreeze;
///
/// struct Foo {
///     nf: NotFreeze
/// }
///
/// fn test<T: Freeze>(t: T) {}
///
/// // This will fail to compile becuase `Foo` is not freezeable
/// # { fn test<T>() {}
/// test(Foo { nf: NotFreeze  })
/// # }
/// ```
#[lang="not_freeze"]
#[cfg(not(stage0))]
#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
pub struct NotFreeze;

#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
#[cfg(stage0)]
#[allow(missing_doc)]
pub struct NotFreeze;

/// A 0-sized struct that is specially recognized as not `Send`.
///
/// Structs and enums containing this value will also not be `Send`
///
/// # Example
///
/// ```rust
/// use std::kinds::NotSend;
///
/// struct Foo {
///     ns: NotSend
/// }
///
/// fn test<T: Send>(t: T) {}
///
/// // This will fail to compile becuase `Foo` is not sendable
/// # { fn test<T>() {}
/// test(Foo { ns: NotSend })
/// # }
/// ```
#[lang="not_send"]
#[cfg(not(stage0))]
#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
pub struct NotSend;

#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
#[cfg(stage0)]
#[allow(missing_doc)]
pub struct NotSend;

/// A non-copyable dummy type.
#[deriving(Eq, TotalEq, Ord, TotalOrd, Clone, DeepClone)]
#[cfg(stage0)]
#[unsafe_no_drop_flag]
pub struct NotPod;

#[cfg(stage0)]
impl ::ops::Drop for NotPod {
    fn drop(&mut self) { }
}

#[cfg(test)]
mod tests {
    use super::{NotPod, NotFreeze, NotSend};
    use util::replace;
    use option::{Some, None};
    use mem::size_of;

    #[test]
    fn test_replace() {
        let mut x = Some(NotPod);
        let y = replace(&mut x, None);
        assert!(x.is_none());
        assert!(y.is_some());
    }

    #[test]
    fn test_sizes() {
        assert_eq!(size_of::<NotPod>(), 0);
        assert_eq!(size_of::<NotFreeze>(), 0);
        assert_eq!(size_of::<NotSend>(), 0);
    }

    #[test]
    fn test_noncopyable() {
        static mut did_run: bool = false;

        struct Foo { five: int }

        impl Drop for Foo {
            fn drop(&mut self) {
                assert_eq!(self.five, 5);
                unsafe {
                    did_run = true;
                }
            }
        }

        {
            let _a = (NotPod, Foo { five: 5 }, NotPod);
        }

        unsafe { assert_eq!(did_run, true); }
    }
}
