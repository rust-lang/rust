/// A macro for triggering an ICE.
/// Calling `bug` instead of panicking will result in a nicer error message and should
/// therefore be preferred over `panic`/`unreachable` or others.
///
/// If you have a span available, you should use [`span_bug`] instead.
///
/// If the bug should only be emitted when compilation didn't fail, [`Session::delay_span_bug`] may be useful.
///
/// [`Session::delay_span_bug`]: rustc_session::Session::delay_span_bug
/// [`span_bug`]: crate::span_bug
#[macro_export]
macro_rules! bug {
    () => ( $crate::bug!("impossible case reached") );
    ($msg:expr) => ({ $crate::util::bug::bug_fmt(::std::format_args!($msg)) });
    ($msg:expr,) => ({ $crate::bug!($msg) });
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::util::bug::bug_fmt(::std::format_args!($fmt, $($arg)+))
    });
}

/// A macro for triggering an ICE with a span.
/// Calling `span_bug!` instead of panicking will result in a nicer error message and point
/// at the code the compiler was compiling when it ICEd. This is the preferred way to trigger
/// ICEs.
///
/// If the bug should only be emitted when compilation didn't fail, [`Session::delay_span_bug`] may be useful.
///
/// [`Session::delay_span_bug`]: rustc_session::Session::delay_span_bug
#[macro_export]
macro_rules! span_bug {
    ($span:expr, $msg:expr) => ({ $crate::util::bug::span_bug_fmt($span, ::std::format_args!($msg)) });
    ($span:expr, $msg:expr,) => ({ $crate::span_bug!($span, $msg) });
    ($span:expr, $fmt:expr, $($arg:tt)+) => ({
        $crate::util::bug::span_bug_fmt($span, ::std::format_args!($fmt, $($arg)+))
    });
}

///////////////////////////////////////////////////////////////////////////
// Lift and TypeFoldable/TypeVisitable macros
//
// When possible, use one of these (relatively) convenient macros to write
// the impls for you.

#[macro_export]
macro_rules! CloneLiftImpls {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'tcx> $crate::ty::Lift<'tcx> for $ty {
                type Lifted = Self;
                fn lift_to_tcx(self, _: $crate::ty::TyCtxt<'tcx>) -> Option<Self> {
                    Some(self)
                }
            }
        )+
    };
}

/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
#[macro_export]
macro_rules! TrivialTypeTraversalImpls {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'tcx> $crate::ty::fold::TypeFoldable<$crate::ty::TyCtxt<'tcx>> for $ty {
                fn try_fold_with<F: $crate::ty::fold::FallibleTypeFolder<$crate::ty::TyCtxt<'tcx>>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::ty::fold::TypeFolder<$crate::ty::TyCtxt<'tcx>>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<'tcx> $crate::ty::visit::TypeVisitable<$crate::ty::TyCtxt<'tcx>> for $ty {
                #[inline]
                fn visit_with<F: $crate::ty::visit::TypeVisitor<$crate::ty::TyCtxt<'tcx>>>(
                    &self,
                    _: &mut F)
                    -> ::std::ops::ControlFlow<F::BreakTy>
                {
                    ::std::ops::ControlFlow::Continue(())
                }
            }
        )+
    };
}

#[macro_export]
macro_rules! TrivialTypeTraversalAndLiftImpls {
    ($($t:tt)*) => {
        TrivialTypeTraversalImpls! { $($t)* }
        CloneLiftImpls! { $($t)* }
    }
}
