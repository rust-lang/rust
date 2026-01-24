//! impl bool {}

use crate::intrinsics;
use crate::ub_checks::assert_unsafe_precondition;

impl bool {
    /// Returns `Some(t)` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// Arguments passed to `then_some` are eagerly evaluated; if you are
    /// passing the result of a function call, it is recommended to use
    /// [`then`], which is lazily evaluated.
    ///
    /// [`then`]: bool::then
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then_some(0), None);
    /// assert_eq!(true.then_some(0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    /// let mut function_with_side_effects = || { a += 1; };
    ///
    /// true.then_some(function_with_side_effects());
    /// false.then_some(function_with_side_effects());
    ///
    /// // `a` is incremented twice because the value passed to `then_some` is
    /// // evaluated eagerly.
    /// assert_eq!(a, 2);
    /// ```
    #[stable(feature = "bool_to_option", since = "1.62.0")]
    #[inline]
    pub fn then_some<T>(self, t: T) -> Option<T> {
        if self { Some(t) } else { None }
    }

    /// Returns `Some(f())` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then(|| 0), None);
    /// assert_eq!(true.then(|| 0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    ///
    /// true.then(|| { a += 1; });
    /// false.then(|| { a += 1; });
    ///
    /// // `a` is incremented once because the closure is evaluated lazily by
    /// // `then`.
    /// assert_eq!(a, 1);
    /// ```
    #[doc(alias = "then_with")]
    #[stable(feature = "lazy_bool_to_option", since = "1.50.0")]
    #[rustc_diagnostic_item = "bool_then"]
    #[inline]
    pub fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self { Some(f()) } else { None }
    }

    /// Returns `Ok(())` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `Err(err)` otherwise.
    ///
    /// Arguments passed to `ok_or` are eagerly evaluated; if you are
    /// passing the result of a function call, it is recommended to use
    /// [`ok_or_else`], which is lazily evaluated.
    ///
    /// [`ok_or_else`]: bool::ok_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bool_to_result)]
    ///
    /// assert_eq!(false.ok_or(0), Err(0));
    /// assert_eq!(true.ok_or(0), Ok(()));
    /// ```
    ///
    /// ```
    /// #![feature(bool_to_result)]
    ///
    /// let mut a = 0;
    /// let mut function_with_side_effects = || { a += 1; };
    ///
    /// assert!(true.ok_or(function_with_side_effects()).is_ok());
    /// assert!(false.ok_or(function_with_side_effects()).is_err());
    ///
    /// // `a` is incremented twice because the value passed to `ok_or` is
    /// // evaluated eagerly.
    /// assert_eq!(a, 2);
    /// ```
    #[unstable(feature = "bool_to_result", issue = "142748")]
    #[inline]
    pub fn ok_or<E>(self, err: E) -> Result<(), E> {
        if self { Ok(()) } else { Err(err) }
    }

    /// Returns `Ok(())` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `Err(f())` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(bool_to_result)]
    ///
    /// assert_eq!(false.ok_or_else(|| 0), Err(0));
    /// assert_eq!(true.ok_or_else(|| 0), Ok(()));
    /// ```
    ///
    /// ```
    /// #![feature(bool_to_result)]
    ///
    /// let mut a = 0;
    ///
    /// assert!(true.ok_or_else(|| { a += 1; }).is_ok());
    /// assert!(false.ok_or_else(|| { a += 1; }).is_err());
    ///
    /// // `a` is incremented once because the closure is evaluated lazily by
    /// // `ok_or_else`.
    /// assert_eq!(a, 1);
    /// ```
    #[unstable(feature = "bool_to_result", issue = "142748")]
    #[inline]
    pub fn ok_or_else<E, F: FnOnce() -> E>(self, f: F) -> Result<(), E> {
        if self { Ok(()) } else { Err(f()) }
    }

    /// Same value as `self | other`, but UB if any bit position is set in both inputs.
    ///
    /// This is a situational micro-optimization for places where you'd rather
    /// use addition on some platforms and bitwise or on other platforms, based
    /// on exactly which instructions combine better with whatever else you're
    /// doing.  Note that there's no reason to bother using this for places
    /// where it's clear from the operations involved that they can't overlap.
    /// For example, if you're combining `u16`s into a `u32` with
    /// `((a as u32) << 16) | (b as u32)`, that's fine, as the backend will
    /// know those sides of the `|` are disjoint without needing help.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(disjoint_bitor)]
    ///
    /// // SAFETY: `false` and `true` have no ones in common.
    /// unsafe { false.unchecked_disjoint_bitor(true) };
    /// ```
    ///
    /// # Safety
    ///
    /// Requires that `self` and `rhs` are disjoint to each other, i.e. do not
    /// have overlapping ones (thus `self & rhs == false`). By extension, requires
    /// that `self | rhs`, `self + rhs`, and `self ^ rhs` are equivalent.
    #[unstable(feature = "disjoint_bitor", issue = "135758")]
    #[rustc_const_unstable(feature = "disjoint_bitor", issue = "135758")]
    #[inline]
    pub const unsafe fn unchecked_disjoint_bitor(self, rhs: Self) -> Self {
        assert_unsafe_precondition!(
            check_language_ub,
            "attempt to disjoint or conjoint values",
            (
                lhs: bool = self,
                rhs: bool = rhs,
            ) => (lhs & rhs) == false,
        );

        // SAFETY: Same precondition.
        unsafe { intrinsics::disjoint_bitor(self, rhs) }
    }
}
