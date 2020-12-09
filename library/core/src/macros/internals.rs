use crate::{fmt, panic};

#[cold]
#[doc(hidden)]
#[unstable(feature = "macros_internals", reason = "macros implementation detail", issue = "none")]
#[track_caller]
pub fn assert_failed<T, U>(op: &str, left: &T, right: &U, args: Option<fmt::Arguments<'_>>) -> !
where
    T: fmt::Debug + ?Sized,
    U: fmt::Debug + ?Sized,
{
    #[track_caller]
    fn inner(
        op: &str,
        left: &dyn fmt::Debug,
        right: &dyn fmt::Debug,
        args: Option<fmt::Arguments<'_>>,
    ) -> ! {
        match args {
            Some(args) => panic!(
                r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}: {}`"#,
                op, left, right, args
            ),
            None => panic!(
                r#"assertion failed: `(left {} right)`
  left: `{:?}`,
 right: `{:?}`"#,
                op, left, right,
            ),
        }
    }
    inner(op, &left, &right, args)
}
