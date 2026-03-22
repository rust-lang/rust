/// Trait for types that support short-circuiting logical operators `&&` and `||`.
///
/// By implementing `Decisive` along with [`BitAnd`] and/or [`BitOr`], a type
/// gains the ability to be used with the `&&` and `||` operators respectively,
/// with proper short-circuit evaluation.
///
/// For `&&`: if `is_false(&lhs)` returns `true`, the right-hand side is not
/// evaluated and the left-hand side is returned directly. Otherwise,
/// `BitAnd::bitand(lhs, rhs)` is evaluated.
///
/// For `||`: if `is_true(&lhs)` returns `true`, the right-hand side is not
/// evaluated and the left-hand side is returned directly. Otherwise,
/// `BitOr::bitor(lhs, rhs)` is evaluated.
///
/// # Three-valued logic example
///
/// This enables behavior trees and other three-valued logic systems:
///
/// ```ignore (illustrative)
/// use std::ops::{BitAnd, BitOr, Decisive};
///
/// #[derive(Copy, Clone)]
/// struct Status(i8);
///
/// const DONE: Status = Status(1);
/// const CONT: Status = Status(0);
/// const FAIL: Status = Status(-1);
///
/// impl Decisive for Status {
///     fn is_true(&self) -> bool { self.0 != -1 }
///     fn is_false(&self) -> bool { self.0 != 1 }
/// }
///
/// impl BitAnd for Status {
///     type Output = Status;
///     fn bitand(self, rhs: Status) -> Status { rhs }
/// }
///
/// impl BitOr for Status {
///     type Output = Status;
///     fn bitor(self, rhs: Status) -> Status { rhs }
/// }
///
/// // Now you can write:
/// // let result = task_a() && task_b() && task_c();  // sequence
/// // let result = task_a() || task_b() || task_c();  // selector
/// ```
#[lang = "decisive"]
#[unstable(feature = "decisive_trait", issue = "none")]
pub trait Decisive {
    /// Returns `true` if this value should short-circuit the `||` operator.
    ///
    /// When used with `||`, if this returns `true` for the left-hand side,
    /// the right-hand side is not evaluated.
    #[unstable(feature = "decisive_trait", issue = "none")]
    fn is_true(&self) -> bool;

    /// Returns `true` if this value should short-circuit the `&&` operator.
    ///
    /// When used with `&&`, if this returns `true` for the left-hand side,
    /// the right-hand side is not evaluated.
    #[unstable(feature = "decisive_trait", issue = "none")]
    fn is_false(&self) -> bool;
}

#[unstable(feature = "decisive_trait", issue = "none")]
impl Decisive for bool {
    #[inline]
    fn is_true(&self) -> bool {
        *self
    }

    #[inline]
    fn is_false(&self) -> bool {
        !*self
    }
}
