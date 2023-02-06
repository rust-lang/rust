/// Used for types that are `Copy` and which **do not care arena
/// allocated data** (i.e., don't need to be folded).
#[macro_export]
macro_rules! TrivialTypeVisitableImpls {
    ($(<$($p:tt),+> $i:ty { $ty:ty } $({where $($wc:tt)+})?)+) => {
        $(
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
        TrivialTypeVisitableImpls! {$(
            <I> I { $ty } {where I: $crate::Interner}
        )+}
    };
}

#[macro_export]
macro_rules! EnumTypeVisitableImpl {
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
                EnumTypeVisitableImpl!(@VisitVariants(self, visitor) input($($variants)*) output())
            }
        }
    };

    (@VisitVariants($this:expr, $visitor:expr) input() output($($output:tt)*)) => {
        match $this {
            $($output)*
        }
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path) ( $($variant_arg:ident),* ) , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeVisitableImpl!(
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
        EnumTypeVisitableImpl!(
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
        EnumTypeVisitableImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant => { ::std::ops::ControlFlow::Continue(()) }
                    $($output)*
                )
        )
    };
}
