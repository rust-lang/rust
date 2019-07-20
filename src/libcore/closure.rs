//! `Closure` trait for examining closure captured variables, and trait implementations for closures
#![stable(feature = "closure_traits", since = "1.38.0")]

use crate::mem::{transmute, transmute_copy, forget};
use crate::fmt::{self, Debug};
use crate::cmp::Ordering;
use crate::hash::{Hash, Hasher};

/// `Closure` is a trait automatically implemented for closures by the compiler
#[stable(feature = "closure_traits", since = "1.38.0")]
#[lang = "closure_trait"]
#[fundamental]
pub unsafe trait Closure: Sized {
    /// The tuple that has equivalent layout to this closure
    #[stable(feature = "closure_traits", since = "1.38.0")]
    type Inner;
}

/// `ClosureExt` is a trait that allows easier use of `Closure`
#[stable(feature = "closure_traits", since = "1.38.0")]
pub trait ClosureExt: Closure {
    /// Get a reference to the tuple that has same layout as this closure
    #[stable(feature = "closure_traits", since = "1.38.0")]
    fn get(&self) -> &Self::Inner;

    /// Get a mutable reference to the tuple that has the same layout as this closure
    #[stable(feature = "closure_traits", since = "1.38.0")]
    fn get_mut(&mut self) -> &mut Self::Inner;

    /// Convert self to the tuple that has same layout as this closure
    #[stable(feature = "closure_traits", since = "1.38.0")]
    fn into_inner(self) -> Self::Inner;

    /// Create a closure from the tuple that has same layout as this closure
    #[stable(feature = "closure_traits", since = "1.38.0")]
    fn from_inner(x: Self::Inner) -> Self;
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> ClosureExt for T {
    fn get(&self) -> &Self::Inner {
        unsafe {transmute(self)}
    }

    fn get_mut(&mut self) -> &mut Self::Inner {
        unsafe {transmute(self)}
    }

    fn into_inner(self) -> Self::Inner {
        let r = unsafe {transmute_copy(&self)};
        forget(self);
        r
    }

    fn from_inner(x: Self::Inner) -> Self {
        let r = unsafe {transmute_copy(&x)};
        forget(x);
        r
    }
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> Debug for T
    where <T as Closure>::Inner: Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // the type name is already printed by the Debug impl for LayoutAs
        Debug::fmt(self.get(), f)
    }
}

// we allow comparisons between versions of the same closure with different
// type parameters, but not between different closures, thanks to the LayoutAs<T>
// marker that has no heterogeneous PartialOrd implementation

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<A: Closure, B: Closure> PartialEq<B> for A
    where <A as Closure>::Inner: PartialEq<<B as Closure>::Inner> {
    #[inline]
    fn eq(&self, other: &B) -> bool { PartialEq::eq(self.get(), other.get()) }
    #[inline]
    fn ne(&self, other: &B) -> bool { PartialEq::ne(self.get(), other.get()) }
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<A: Closure, B: Closure> PartialOrd<B> for A
    where <A as Closure>::Inner: PartialOrd<<B as Closure>::Inner> {
    #[inline]
    fn partial_cmp(&self, other: &B) -> Option<Ordering> {
        PartialOrd::partial_cmp(self.get(), other.get())
    }
    #[inline]
    fn lt(&self, other: &B) -> bool { PartialOrd::lt(self.get(), other.get()) }
    #[inline]
    fn le(&self, other: &B) -> bool { PartialOrd::le(self.get(), other.get()) }
    #[inline]
    fn ge(&self, other: &B) -> bool { PartialOrd::ge(self.get(), other.get()) }
    #[inline]
    fn gt(&self, other: &B) -> bool { PartialOrd::gt(self.get(), other.get()) }
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> Ord for T
    where <T as Closure>::Inner: Ord {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering { Ord::cmp(self.get(), other.get()) }
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> Eq for T
    where <T as Closure>::Inner: Eq {
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> Hash for T
    where <T as Closure>::Inner: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

#[stable(feature = "closure_traits", since = "1.38.0")]
impl<T: Closure> Default for T
    where <T as Closure>::Inner: Default {
    fn default() -> Self {
        <T as ClosureExt>::from_inner(Default::default())
    }
}
