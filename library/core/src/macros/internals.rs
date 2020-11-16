use crate::{fmt, panic};

#[cold]
#[doc(hidden)]
#[unstable(feature = "macros_internals", reason = "macros implementation detail", issue = "none")]
#[track_caller]
pub fn assert_eq_failed<T, U>(left: &T, right: &U) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    #[track_caller]
    fn inner(left: &dyn fmt::Debug, right: &dyn fmt::Debug) -> ! {
        panic!(
            r#"assertion failed: `(left == right)`
left: `{:?}`,
right: `{:?}`"#,
            left, right
        )
    }
    inner(&left, &right)
}

#[cold]
#[doc(hidden)]
#[unstable(feature = "macros_internals", reason = "macros implementation detail", issue = "none")]
#[track_caller]
pub fn assert_eq_failed_args<T, U>(left: &T, right: &U, args: fmt::Arguments<'_>) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    #[track_caller]
    fn inner(left: &dyn fmt::Debug, right: &dyn fmt::Debug, args: fmt::Arguments<'_>) -> ! {
        panic!(
            r#"assertion failed: `(left == right)`
left: `{:?}`,
right: `{:?}: {}`"#,
            left, right, args
        )
    }
    inner(&left, &right, args)
}

#[cold]
#[doc(hidden)]
#[unstable(feature = "macros_internals", reason = "macros implementation detail", issue = "none")]
#[track_caller]
pub fn assert_ne_failed<T, U>(left: &T, right: &U) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    #[track_caller]
    fn inner(left: &dyn fmt::Debug, right: &dyn fmt::Debug) -> ! {
        panic!(
            r#"assertion failed: `(left != right)`
left: `{:?}`,
right: `{:?}`"#,
            left, right
        )
    }
    inner(&left, &right)
}

#[cold]
#[doc(hidden)]
#[unstable(feature = "macros_internals", reason = "macros implementation detail", issue = "none")]
#[track_caller]
pub fn assert_ne_failed_args<T, U>(left: &T, right: &U, args: fmt::Arguments<'_>) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    #[track_caller]
    fn inner(left: &dyn fmt::Debug, right: &dyn fmt::Debug, args: fmt::Arguments<'_>) -> ! {
        panic!(
            r#"assertion failed: `(left != right)`
left: `{:?}`,
right: `{:?}: {}`"#,
            left, right, args
        )
    }
    inner(&left, &right, args)
}
