use crate::convert::Infallible;
use crate::iter::FromIterator;
use crate::ops::{ControlFlow, FromResidual, Try};

/// Collapses all unit items from an iterator into one.
///
/// This is more useful when combined with higher-level abstractions, like
/// collecting to a `Result<(), E>` where you only care about errors:
///
/// ```
/// use std::io::*;
/// let data = vec![1, 2, 3, 4, 5];
/// let res: Result<()> = data.iter()
///     .map(|x| writeln!(stdout(), "{x}"))
///     .collect();
/// assert!(res.is_ok());
/// ```
#[stable(feature = "unit_from_iter", since = "1.23.0")]
impl FromIterator<()> for () {
    fn from_iter<I: IntoIterator<Item = ()>>(iter: I) -> Self {
        iter.into_iter().for_each(|()| {})
    }
}

/// Allows the unit type to be used with the question mark operator.
///
/// This allows more easily writing internal iteration functions which can take
/// both fallible and infallible closures, or visitors which can have both
/// fallible and infallible implementations.
///
/// # Examples
///
/// ```
/// #![feature(try_trait_v2)]
///
/// fn foreach<I, R>(iter: I, mut f: impl FnMut(I::Item) -> R) -> R
/// where
///     I: IntoIterator,
///     R: std::ops::Try<Output = ()>,
/// {
///     for x in iter {
///         f(x)?;
///     }
///     R::from_output(())
/// }
///
/// // prints everything
/// foreach([1, 2, 3], |x| println!("{x}"));
///
/// // prints everything until an error occurs
/// let _: Result<_, ()> = foreach([Ok(1), Ok(2), Err(()), Ok(3)], |x| {
///     println!("{}", x?);
///     Ok(())
/// });
/// ```
///
/// ```
/// #![feature(try_trait_v2)]
///
/// fn walk_children<V: Visitor>(visitor: &mut V, item: ()) -> V::ResultTy {
///     // can use the `?` operator if needed here.
///     <<V as Visitor>::ResultTy as std::ops::Try>::from_output(())
/// }
///
/// trait Visitor: Sized {
///     type ResultTy: std::ops::Try<Output = ()>;
///
///     fn visit_x(&mut self, item: ()) -> Self::ResultTy {
///         // can use the `?` operator if needed here.
///         walk_children(self, item)
///     }
///     // some `visit_*` functions
/// }
///
/// struct InfallibleVisitor;
///
/// impl Visitor for InfallibleVisitor {
///     type ResultTy = ();
///     // implement `visit_*` functions
/// }
///
/// struct FallibleVisitor;
///
/// impl Visitor for FallibleVisitor {
///     type ResultTy = std::ops::ControlFlow<()>;
///     // implement `visit_*` functions
/// }
/// ```
#[unstable(feature = "try_trait_v2", issue = "84277")]
impl Try for () {
    type Output = ();
    type Residual = Infallible;

    fn from_output(_: ()) -> Self {
        ()
    }

    fn branch(self) -> ControlFlow<Infallible, ()> {
        ControlFlow::Continue(())
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl FromResidual<Infallible> for () {
    fn from_residual(_: Infallible) -> Self {
        ()
    }
}
