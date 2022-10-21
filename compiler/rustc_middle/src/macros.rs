#[macro_export]
macro_rules! bug {
    () => ( $crate::bug!("impossible case reached") );
    ($msg:expr) => ({ $crate::util::bug::bug_fmt(::std::format_args!($msg)) });
    ($msg:expr,) => ({ $crate::bug!($msg) });
    ($fmt:expr, $($arg:tt)+) => ({
        $crate::util::bug::bug_fmt(::std::format_args!($fmt, $($arg)+))
    });
}

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
    (for <$tcx:lifetime> { $($ty:ty,)+ }) => {
        $(
            impl<$tcx> $crate::ty::Lift<$tcx> for $ty {
                type Lifted = Self;
                fn lift_to_tcx(self, _: $crate::ty::TyCtxt<$tcx>) -> Option<Self> {
                    Some(self)
                }
            }
        )+
    };

    ($($ty:ty,)+) => {
        CloneLiftImpls! {
            for <'tcx> {
                $($ty,)+
            }
        }
    };
}

/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
#[macro_export]
macro_rules! TrivialTypeTraversalImpls {
    (for <$tcx:lifetime> { $($ty:ty,)+ }) => {
        $(
            impl<$tcx> $crate::ty::fold::TypeFoldable<$tcx> for $ty {
                fn try_fold_with<F: $crate::ty::fold::FallibleTypeFolder<$tcx>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::ty::fold::TypeFolder<$tcx>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<$tcx> $crate::ty::visit::TypeVisitable<$tcx> for $ty {
                #[inline]
                fn visit_with<F: $crate::ty::visit::TypeVisitor<$tcx>>(
                    &self,
                    _: &mut F)
                    -> ::std::ops::ControlFlow<F::BreakTy>
                {
                    ::std::ops::ControlFlow::CONTINUE
                }
            }
        )+
    };

    ($($ty:ty,)+) => {
        TrivialTypeTraversalImpls! {
            for <'tcx> {
                $($ty,)+
            }
        }
    };
}

#[macro_export]
macro_rules! TrivialTypeTraversalAndLiftImpls {
    ($($t:tt)*) => {
        TrivialTypeTraversalImpls! { $($t)* }
        CloneLiftImpls! { $($t)* }
    }
}

#[macro_export]
macro_rules! EnumTypeTraversalImpl {
    (impl<$($p:tt),*> TypeFoldable<$tcx:tt> for $s:path {
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::fold::TypeFoldable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn try_fold_with<V: $crate::ty::fold::FallibleTypeFolder<$tcx>>(
                self,
                folder: &mut V,
            ) -> ::std::result::Result<Self, V::Error> {
                EnumTypeTraversalImpl!(@FoldVariants(self, folder) input($($variants)*) output())
            }
        }
    };

    (impl<$($p:tt),*> TypeVisitable<$tcx:tt> for $s:path {
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::visit::TypeVisitable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn visit_with<V: $crate::ty::visit::TypeVisitor<$tcx>>(
                &self,
                visitor: &mut V,
            ) -> ::std::ops::ControlFlow<V::BreakTy> {
                EnumTypeTraversalImpl!(@VisitVariants(self, visitor) input($($variants)*) output())
            }
        }
    };

    (@FoldVariants($this:expr, $folder:expr) input() output($($output:tt)*)) => {
        Ok(match $this {
            $($output)*
        })
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path) ( $($variant_arg:ident),* ) , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant ( $($variant_arg),* ) => {
                        $variant (
                            $($crate::ty::fold::TypeFoldable::try_fold_with($variant_arg, $folder)?),*
                        )
                    }
                    $($output)*
                )
        )
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path) { $($variant_arg:ident),* $(,)? } , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant { $($variant_arg),* } => {
                        $variant {
                            $($variant_arg: $crate::ty::fold::TypeFoldable::fold_with(
                                $variant_arg, $folder
                            )?),* }
                    }
                    $($output)*
                )
        )
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path), $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant => { $variant }
                    $($output)*
                )
        )
    };

    (@VisitVariants($this:expr, $visitor:expr) input() output($($output:tt)*)) => {
        match $this {
            $($output)*
        }
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path) ( $($variant_arg:ident),* ) , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant ( $($variant_arg),* ) => {
                        $($crate::ty::visit::TypeVisitable::visit_with(
                            $variant_arg, $visitor
                        )?;)*
                        ::std::ops::ControlFlow::CONTINUE
                    }
                    $($output)*
                )
        )
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path) { $($variant_arg:ident),* $(,)? } , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant { $($variant_arg),* } => {
                        $($crate::ty::visit::TypeVisitable::visit_with(
                            $variant_arg, $visitor
                        )?;)*
                        ::std::ops::ControlFlow::CONTINUE
                    }
                    $($output)*
                )
        )
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path), $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeTraversalImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant => { ::std::ops::ControlFlow::CONTINUE }
                    $($output)*
                )
        )
    };
}
