/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
#[macro_export]
macro_rules! TrivialTypeTraversalImpls {
    ($(<$($p:tt),+> $i:ty { $ty:ty } $({where $($wc:tt)+})?)+) => {
        $(
            impl<$($p),+> $crate::fold::TypeFoldable<$i> for $ty $(where $($wc)+)? {
                fn try_fold_with<F: $crate::fold::FallibleTypeFolder<$i>>(
                    self,
                    _: &mut F,
                ) -> ::std::result::Result<Self, F::Error> {
                    Ok(self)
                }

                #[inline]
                fn fold_with<F: $crate::fold::TypeFolder<$i>>(
                    self,
                    _: &mut F,
                ) -> Self {
                    self
                }
            }

            impl<$($p),+> $crate::visit::TypeVisitable<$i> for $ty $(where $($wc)+)? {
                #[inline]
                fn visit_with<F: $crate::visit::TypeVisitor<$i>>(
                    &self,
                    _: &mut F)
                    -> ::std::ops::ControlFlow<F::BreakTy>
                {
                    ::std::ops::ControlFlow::Continue(())
                }
            }
        )+
    };

    ($($ty:ty,)+) => {
        TrivialTypeTraversalImpls! {$(
            <I> I { $ty } {where I: $crate::Interner}
        )+}
    };
}

#[macro_export]
macro_rules! EnumTypeTraversalImpl {
    (impl<$($p:tt),*> TypeFoldable<$tcx:ty> for $s:path {
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::fold::TypeFoldable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn try_fold_with<V: $crate::fold::FallibleTypeFolder<$tcx>>(
                self,
                folder: &mut V,
            ) -> ::std::result::Result<Self, V::Error> {
                EnumTypeTraversalImpl!(@FoldVariants(self, folder) input($($variants)*) output())
            }
        }
    };

    (impl<$($p:tt),*> TypeVisitable<$tcx:ty> for $s:path {
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::visit::TypeVisitable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn visit_with<V: $crate::visit::TypeVisitor<$tcx>>(
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
                            $($crate::fold::TypeFoldable::try_fold_with($variant_arg, $folder)?),*
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
                            $($variant_arg: $crate::fold::TypeFoldable::fold_with(
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
                        $($crate::visit::TypeVisitable::visit_with(
                            $variant_arg, $visitor
                        )?;)*
                        ::std::ops::ControlFlow::Continue(())
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
                        $($crate::visit::TypeVisitable::visit_with(
                            $variant_arg, $visitor
                        )?;)*
                        ::std::ops::ControlFlow::Continue(())
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
                    $variant => { ::std::ops::ControlFlow::Continue(()) }
                    $($output)*
                )
        )
    };
}
