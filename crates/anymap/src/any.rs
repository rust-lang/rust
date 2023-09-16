//! Copyright © 2014–2022 Chris Morgan
//! https://github.com/chris-morgan/anymap/blob/master/COPYING
//! impl some traits for dyn Any
use core::any::{Any, TypeId};
use core::fmt;

#[doc(hidden)]
pub trait CloneToAny {
    /// Clone `self` into a new `Box<dyn CloneAny>` object.
    fn clone_to_any(&self) -> Box<dyn CloneAny>;
}

impl<T: Any + Clone> CloneToAny for T {
    #[inline]
    fn clone_to_any(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }
}

macro_rules! impl_clone {
    ($t:ty) => {
        impl Clone for Box<$t> {
            #[inline]
            fn clone(&self) -> Box<$t> {
                // SAFETY: this dance is to reapply any Send/Sync marker. I’m not happy about this
                // approach, given that I used to do it in safe code, but then came a dodgy
                // future-compatibility warning where_clauses_object_safety, which is spurious for
                // auto traits but still super annoying (future-compatibility lints seem to mean
                // your bin crate needs a corresponding allow!). Although I explained my plight¹
                // and it was all explained and agreed upon, no action has been taken. So I finally
                // caved and worked around it by doing it this way, which matches what’s done for
                // core::any², so it’s probably not *too* bad.
                //
                // ¹ https://github.com/rust-lang/rust/issues/51443#issuecomment-421988013
                // ² https://github.com/rust-lang/rust/blob/e7825f2b690c9a0d21b6f6d84c404bb53b151b38/library/alloc/src/boxed.rs#L1613-L1616
                let clone: Box<dyn CloneAny> = (**self).clone_to_any();
                let raw: *mut dyn CloneAny = Box::into_raw(clone);
                unsafe { Box::from_raw(raw as *mut $t) }
            }
        }

        impl fmt::Debug for $t {
            #[inline]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.pad(stringify!($t))
            }
        }
    };
}

/// Methods for downcasting from an `Any`-like trait object.
///
/// This should only be implemented on trait objects for subtraits of `Any`, though you can
/// implement it for other types and it’ll work fine, so long as your implementation is correct.
pub trait Downcast {
    /// Gets the `TypeId` of `self`.
    fn type_id(&self) -> TypeId;

    // Note the bound through these downcast methods is 'static, rather than the inexpressible
    // concept of Self-but-as-a-trait (where Self is `dyn Trait`). This is sufficient, exceeding
    // TypeId’s requirements. Sure, you *can* do CloneAny.downcast_unchecked::<NotClone>() and the
    // type system won’t protect you, but that doesn’t introduce any unsafety: the method is
    // already unsafe because you can specify the wrong type, and if this were exposing safe
    // downcasting, CloneAny.downcast::<NotClone>() would just return an error, which is just as
    // correct.
    //
    // Now in theory we could also add T: ?Sized, but that doesn’t play nicely with the common
    // implementation, so I’m doing without it.

    /// Downcast from `&Any` to `&T`, without checking the type matches.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `T` matches the trait object, on pain of *undefined behaviour*.
    unsafe fn downcast_ref_unchecked<T: 'static>(&self) -> &T;

    /// Downcast from `&mut Any` to `&mut T`, without checking the type matches.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `T` matches the trait object, on pain of *undefined behaviour*.
    unsafe fn downcast_mut_unchecked<T: 'static>(&mut self) -> &mut T;
}

/// A trait for the conversion of an object into a boxed trait object.
pub trait IntoBox<A: ?Sized + Downcast>: Any {
    /// Convert self into the appropriate boxed form.
    fn into_box(self) -> Box<A>;
}

macro_rules! implement {
    ($any_trait:ident $(+ $auto_traits:ident)*) => {
        impl Downcast for dyn $any_trait $(+ $auto_traits)* {
            #[inline]
            fn type_id(&self) -> TypeId {
                self.type_id()
            }

            #[inline]
            unsafe fn downcast_ref_unchecked<T: 'static>(&self) -> &T {
                &*(self as *const Self as *const T)
            }

            #[inline]
            unsafe fn downcast_mut_unchecked<T: 'static>(&mut self) -> &mut T {
                &mut *(self as *mut Self as *mut T)
            }
        }

        impl<T: $any_trait $(+ $auto_traits)*> IntoBox<dyn $any_trait $(+ $auto_traits)*> for T {
            #[inline]
            fn into_box(self) -> Box<dyn $any_trait $(+ $auto_traits)*> {
                Box::new(self)
            }
        }
    }
}

implement!(Any);
implement!(Any + Send);
implement!(Any + Send + Sync);

/// [`Any`], but with cloning.
///
/// Every type with no non-`'static` references that implements `Clone` implements `CloneAny`.
/// See [`core::any`] for more details on `Any` in general.
pub trait CloneAny: Any + CloneToAny {}
impl<T: Any + Clone> CloneAny for T {}
implement!(CloneAny);
implement!(CloneAny + Send);
implement!(CloneAny + Send + Sync);
impl_clone!(dyn CloneAny);
impl_clone!(dyn CloneAny + Send);
impl_clone!(dyn CloneAny + Send + Sync);
