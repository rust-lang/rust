use crate::{convert, ops};

/// Used to tell an operation whether it should exit early or go on as usual.
///
/// This is used when exposing things (like graph traversals or visitors) where
/// you want the user to be able to choose whether to exit early.
/// Having the enum makes it clearer -- no more wondering "wait, what did `false`
/// mean again?" -- and allows including a value.
///
/// Similar to [`Option`] and [`Result`], this enum can be used with the `?` operator
/// to return immediately if the [`Break`] variant is present or otherwise continue normally
/// with the value inside the [`Continue`] variant.
///
/// # Examples
///
/// Early-exiting from [`Iterator::try_for_each`]:
/// ```
/// use std::ops::ControlFlow;
///
/// let r = (2..100).try_for_each(|x| {
///     if 403 % x == 0 {
///         return ControlFlow::Break(x)
///     }
///
///     ControlFlow::Continue(())
/// });
/// assert_eq!(r, ControlFlow::Break(13));
/// ```
///
/// A basic tree traversal:
/// ```
/// use std::ops::ControlFlow;
///
/// pub struct TreeNode<T> {
///     value: T,
///     left: Option<Box<TreeNode<T>>>,
///     right: Option<Box<TreeNode<T>>>,
/// }
///
/// impl<T> TreeNode<T> {
///     pub fn traverse_inorder<B>(&self, f: &mut impl FnMut(&T) -> ControlFlow<B>) -> ControlFlow<B> {
///         if let Some(left) = &self.left {
///             left.traverse_inorder(f)?;
///         }
///         f(&self.value)?;
///         if let Some(right) = &self.right {
///             right.traverse_inorder(f)?;
///         }
///         ControlFlow::Continue(())
///     }
///     fn leaf(value: T) -> Option<Box<TreeNode<T>>> {
///         Some(Box::new(Self { value, left: None, right: None }))
///     }
/// }
///
/// let node = TreeNode {
///     value: 0,
///     left: TreeNode::leaf(1),
///     right: Some(Box::new(TreeNode {
///         value: -1,
///         left: TreeNode::leaf(5),
///         right: TreeNode::leaf(2),
///     }))
/// };
/// let mut sum = 0;
///
/// let res = node.traverse_inorder(&mut |val| {
///     if *val < 0 {
///         ControlFlow::Break(*val)
///     } else {
///         sum += *val;
///         ControlFlow::Continue(())
///     }
/// });
/// assert_eq!(res, ControlFlow::Break(-1));
/// assert_eq!(sum, 6);
/// ```
///
/// [`Break`]: ControlFlow::Break
/// [`Continue`]: ControlFlow::Continue
#[stable(feature = "control_flow_enum_type", since = "1.55.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "ControlFlow")]
// ControlFlow should not implement PartialOrd or Ord, per RFC 3058:
// https://rust-lang.github.io/rfcs/3058-try-trait-v2.html#traits-for-controlflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ControlFlow<B, C = ()> {
    /// Move on to the next phase of the operation as normal.
    #[stable(feature = "control_flow_enum_type", since = "1.55.0")]
    #[lang = "Continue"]
    Continue(C),
    /// Exit the operation without running subsequent phases.
    #[stable(feature = "control_flow_enum_type", since = "1.55.0")]
    #[lang = "Break"]
    Break(B),
    // Yes, the order of the variants doesn't match the type parameters.
    // They're in this order so that `ControlFlow<A, B>` <-> `Result<B, A>`
    // is a no-op conversion in the `Try` implementation.
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<B, C> ops::Try for ControlFlow<B, C> {
    type Output = C;
    type Residual = ControlFlow<B, convert::Infallible>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        ControlFlow::Continue(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            ControlFlow::Continue(c) => ControlFlow::Continue(c),
            ControlFlow::Break(b) => ControlFlow::Break(ControlFlow::Break(b)),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
// Note: manually specifying the residual type instead of using the default to work around
// https://github.com/rust-lang/rust/issues/99940
impl<B, C> ops::FromResidual<ControlFlow<B, convert::Infallible>> for ControlFlow<B, C> {
    #[inline]
    fn from_residual(residual: ControlFlow<B, convert::Infallible>) -> Self {
        match residual {
            ControlFlow::Break(b) => ControlFlow::Break(b),
        }
    }
}

#[unstable(feature = "try_trait_v2_residual", issue = "91285")]
impl<B, C> ops::Residual<C> for ControlFlow<B, convert::Infallible> {
    type TryType = ControlFlow<B, C>;
}

impl<B, C> ControlFlow<B, C> {
    /// Returns `true` if this is a `Break` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// assert!(ControlFlow::<&str, i32>::Break("Stop right there!").is_break());
    /// assert!(!ControlFlow::<&str, i32>::Continue(3).is_break());
    /// ```
    #[inline]
    #[stable(feature = "control_flow_enum_is", since = "1.59.0")]
    pub fn is_break(&self) -> bool {
        matches!(*self, ControlFlow::Break(_))
    }

    /// Returns `true` if this is a `Continue` variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// assert!(!ControlFlow::<&str, i32>::Break("Stop right there!").is_continue());
    /// assert!(ControlFlow::<&str, i32>::Continue(3).is_continue());
    /// ```
    #[inline]
    #[stable(feature = "control_flow_enum_is", since = "1.59.0")]
    pub fn is_continue(&self) -> bool {
        matches!(*self, ControlFlow::Continue(_))
    }

    /// Converts the `ControlFlow` into an `Option` which is `Some` if the
    /// `ControlFlow` was `Break` and `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// assert_eq!(ControlFlow::<&str, i32>::Break("Stop right there!").break_value(), Some("Stop right there!"));
    /// assert_eq!(ControlFlow::<&str, i32>::Continue(3).break_value(), None);
    /// ```
    #[inline]
    #[stable(feature = "control_flow_enum", since = "1.83.0")]
    pub fn break_value(self) -> Option<B> {
        match self {
            ControlFlow::Continue(..) => None,
            ControlFlow::Break(x) => Some(x),
        }
    }

    /// Maps `ControlFlow<B, C>` to `ControlFlow<T, C>` by applying a function
    /// to the break value in case it exists.
    #[inline]
    #[stable(feature = "control_flow_enum", since = "1.83.0")]
    pub fn map_break<T>(self, f: impl FnOnce(B) -> T) -> ControlFlow<T, C> {
        match self {
            ControlFlow::Continue(x) => ControlFlow::Continue(x),
            ControlFlow::Break(x) => ControlFlow::Break(f(x)),
        }
    }

    /// Converts the `ControlFlow` into an `Option` which is `Some` if the
    /// `ControlFlow` was `Continue` and `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ops::ControlFlow;
    ///
    /// assert_eq!(ControlFlow::<&str, i32>::Break("Stop right there!").continue_value(), None);
    /// assert_eq!(ControlFlow::<&str, i32>::Continue(3).continue_value(), Some(3));
    /// ```
    #[inline]
    #[stable(feature = "control_flow_enum", since = "1.83.0")]
    pub fn continue_value(self) -> Option<C> {
        match self {
            ControlFlow::Continue(x) => Some(x),
            ControlFlow::Break(..) => None,
        }
    }

    /// Maps `ControlFlow<B, C>` to `ControlFlow<B, T>` by applying a function
    /// to the continue value in case it exists.
    #[inline]
    #[stable(feature = "control_flow_enum", since = "1.83.0")]
    pub fn map_continue<T>(self, f: impl FnOnce(C) -> T) -> ControlFlow<B, T> {
        match self {
            ControlFlow::Continue(x) => ControlFlow::Continue(f(x)),
            ControlFlow::Break(x) => ControlFlow::Break(x),
        }
    }
}

impl<T> ControlFlow<T, T> {
    /// Extracts the value `T` that is wrapped by `ControlFlow<T, T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(control_flow_into_value)]
    /// use std::ops::ControlFlow;
    ///
    /// assert_eq!(ControlFlow::<i32, i32>::Break(1024).into_value(), 1024);
    /// assert_eq!(ControlFlow::<i32, i32>::Continue(512).into_value(), 512);
    /// ```
    #[unstable(feature = "control_flow_into_value", issue = "137461")]
    #[rustc_allow_const_fn_unstable(const_precise_live_drops)]
    pub const fn into_value(self) -> T {
        match self {
            ControlFlow::Continue(x) | ControlFlow::Break(x) => x,
        }
    }
}

/// These are used only as part of implementing the iterator adapters.
/// They have mediocre names and non-obvious semantics, so aren't
/// currently on a path to potential stabilization.
impl<R: ops::Try> ControlFlow<R, R::Output> {
    /// Creates a `ControlFlow` from any type implementing `Try`.
    #[inline]
    pub(crate) fn from_try(r: R) -> Self {
        match R::branch(r) {
            ControlFlow::Continue(v) => ControlFlow::Continue(v),
            ControlFlow::Break(v) => ControlFlow::Break(R::from_residual(v)),
        }
    }

    /// Converts a `ControlFlow` into any type implementing `Try`.
    #[inline]
    pub(crate) fn into_try(self) -> R {
        match self {
            ControlFlow::Continue(v) => R::from_output(v),
            ControlFlow::Break(v) => v,
        }
    }
}
