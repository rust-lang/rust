// ignore-tidy-linelength

macro_rules! enum_from_u32 {
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident = $e:expr,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant = $e),*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    };
    ($(#[$attr:meta])* pub enum $name:ident {
        $($variant:ident,)*
    }) => {
        $(#[$attr])*
        pub enum $name {
            $($variant,)*
        }

        impl $name {
            pub fn from_u32(u: u32) -> Option<$name> {
                $(if u == $name::$variant as u32 {
                    return Some($name::$variant)
                })*
                None
            }
        }
    }
}

#[macro_export]
macro_rules! bug {
    () => ( bug!("impossible case reached") );
    ($($message:tt)*) => ({
        $crate::util::bug::bug_fmt(file!(), line!(), format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! span_bug {
    ($span:expr, $($message:tt)*) => ({
        $crate::util::bug::span_bug_fmt(file!(), line!(), $span, format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! __impl_stable_hash_field {
    ($field:ident, $ctx:expr, $hasher:expr) => ($field.hash_stable($ctx, $hasher));
    ($field:ident, $ctx:expr, $hasher:expr, _) => ({ let _ = $field; });
    ($field:ident, $ctx:expr, $hasher:expr, $delegate:expr) => ($delegate.hash_stable($ctx, $hasher));
}

#[macro_export]
macro_rules! impl_stable_hash_for {
    // Enums
    (enum $enum_name:path {
        $( $variant:ident
           // this incorrectly allows specifying both tuple-like and struct-like fields, as in `Variant(a,b){c,d}`,
           // when it should be only one or the other
           $( ( $($field:ident $(-> $delegate:tt)?),* ) )?
           $( { $($named_field:ident $(-> $named_delegate:tt)?),* } )?
        ),* $(,)?
    }) => {
        impl_stable_hash_for!(
            impl<> for enum $enum_name [ $enum_name ] { $( $variant
                $( ( $($field $(-> $delegate)?),* ) )?
                $( { $($named_field $(-> $named_delegate)?),* } )?
            ),* }
        );
    };
    // We want to use the enum name both in the `impl ... for $enum_name` as well as for
    // importing all the variants. Unfortunately it seems we have to take the name
    // twice for this purpose
    (impl<$($lt:lifetime $(: $lt_bound:lifetime)? ),* $(,)? $($T:ident),* $(,)?>
        for enum $enum_name:path
        [ $enum_path:path ]
    {
        $( $variant:ident
           // this incorrectly allows specifying both tuple-like and struct-like fields, as in `Variant(a,b){c,d}`,
           // when it should be only one or the other
           $( ( $($field:ident $(-> $delegate:tt)?),* ) )?
           $( { $($named_field:ident $(-> $named_delegate:tt)?),* } )?
        ),* $(,)?
    }) => {
        impl<'a, $($lt $(: $lt_bound)?,)* $($T,)*>
            ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>>
            for $enum_name
            where $($T: ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>>),*
        {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                use $enum_path::*;
                ::std::mem::discriminant(self).hash_stable(__ctx, __hasher);

                match *self {
                    $(
                        $variant $( ( $(ref $field),* ) )? $( { $(ref $named_field),* } )? => {
                            $($( __impl_stable_hash_field!($field, __ctx, __hasher $(, $delegate)?) );*)?
                            $($( __impl_stable_hash_field!($named_field, __ctx, __hasher $(, $named_delegate)?) );*)?
                        }
                    )*
                }
            }
        }
    };
    // Structs
    (struct $struct_name:path { $($field:ident $(-> $delegate:tt)?),* $(,)? }) => {
        impl_stable_hash_for!(
            impl<'tcx> for struct $struct_name { $($field $(-> $delegate)?),* }
        );
    };
    (impl<$($lt:lifetime $(: $lt_bound:lifetime)? ),* $(,)? $($T:ident),* $(,)?> for struct $struct_name:path {
        $($field:ident $(-> $delegate:tt)?),* $(,)?
    }) => {
        impl<'a, $($lt $(: $lt_bound)?,)* $($T,)*>
            ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>> for $struct_name
            where $($T: ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>>),*
        {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                let $struct_name {
                    $(ref $field),*
                } = *self;

                $( __impl_stable_hash_field!($field, __ctx, __hasher $(, $delegate)?) );*
            }
        }
    };
    // Tuple structs
    // We cannot use normal parentheses here, the parser won't allow it
    (tuple_struct $struct_name:path { $($field:ident $(-> $delegate:tt)?),*  $(,)? }) => {
        impl_stable_hash_for!(
            impl<'tcx> for tuple_struct $struct_name { $($field $(-> $delegate)?),* }
        );
    };
    (impl<$($lt:lifetime $(: $lt_bound:lifetime)? ),* $(,)? $($T:ident),* $(,)?>
     for tuple_struct $struct_name:path { $($field:ident $(-> $delegate:tt)?),*  $(,)? }) => {
        impl<'a, $($lt $(: $lt_bound)?,)* $($T,)*>
            ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>> for $struct_name
            where $($T: ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a>>),*
        {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                let $struct_name (
                    $(ref $field),*
                ) = *self;

                $( __impl_stable_hash_field!($field, __ctx, __hasher $(, $delegate)?) );*
            }
        }
    };
}

#[macro_export]
macro_rules! impl_stable_hash_for_spanned {
    ($T:path) => (

        impl<'a, 'tcx> HashStable<StableHashingContext<'a>> for ::syntax::source_map::Spanned<$T>
        {
            #[inline]
            fn hash_stable<W: StableHasherResult>(&self,
                                                  hcx: &mut StableHashingContext<'a>,
                                                  hasher: &mut StableHasher<W>) {
                self.node.hash_stable(hcx, hasher);
                self.span.hash_stable(hcx, hasher);
            }
        }
    );
}

///////////////////////////////////////////////////////////////////////////
// Lift and TypeFoldable macros
//
// When possible, use one of these (relatively) convenient macros to write
// the impls for you.

#[macro_export]
macro_rules! CloneLiftImpls {
    (for <$tcx:lifetime> { $($ty:ty,)+ }) => {
        $(
            impl<$tcx> $crate::ty::Lift<$tcx> for $ty {
                type Lifted = Self;
                fn lift_to_tcx<'a, 'gcx>(&self, _: $crate::ty::TyCtxt<'a, 'gcx, $tcx>) -> Option<Self> {
                    Some(Clone::clone(self))
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
macro_rules! CloneTypeFoldableImpls {
    (for <$tcx:lifetime> { $($ty:ty,)+ }) => {
        $(
            impl<$tcx> $crate::ty::fold::TypeFoldable<$tcx> for $ty {
                fn super_fold_with<'gcx: $tcx, F: $crate::ty::fold::TypeFolder<'gcx, $tcx>>(
                    &self,
                    _: &mut F
                ) -> Result<$ty, F::Error> {
                    Ok(Clone::clone(self))
                }

                fn super_visit_with<F: $crate::ty::fold::TypeVisitor<$tcx>>(
                    &self,
                    _: &mut F)
                    -> Result<(), F::Error>
                {
                    Ok(())
                }
            }
        )+
    };

    ($($ty:ty,)+) => {
        CloneTypeFoldableImpls! {
            for <'tcx> {
                $($ty,)+
            }
        }
    };
}

#[macro_export]
macro_rules! CloneTypeFoldableAndLiftImpls {
    ($($t:tt)*) => {
        CloneTypeFoldableImpls! { $($t)* }
        CloneLiftImpls! { $($t)* }
    }
}

#[macro_export]
macro_rules! BraceStructLiftImpl {
    (impl<$($p:tt),*> Lift<$tcx:tt> for $s:path {
        type Lifted = $lifted:ty;
        $($field:ident),* $(,)?
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::Lift<$tcx> for $s
            $(where $($wc)*)*
        {
            type Lifted = $lifted;

            fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<$lifted> {
                $(let $field = tcx.lift(&self.$field)?;)*
                Some(Self::Lifted { $($field),* })
            }
        }
    };
}

#[macro_export]
macro_rules! EnumLiftImpl {
    (impl<$($p:tt),*> Lift<$tcx:tt> for $s:path {
        type Lifted = $lifted:ty;
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::Lift<$tcx> for $s
            $(where $($wc)*)*
        {
            type Lifted = $lifted;

            fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<$lifted> {
                EnumLiftImpl!(@Variants(self, tcx) input($($variants)*) output())
            }
        }
    };

    (@Variants($this:expr, $tcx:expr) input() output($($output:tt)*)) => {
        match $this {
            $($output)*
        }
    };

    (@Variants($this:expr, $tcx:expr)
     input( ($variant:path) ( $($variant_arg:ident),* ) , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumLiftImpl!(
            @Variants($this, $tcx)
                input($($input)*)
                output(
                    $variant ( $($variant_arg),* ) => {
                        Some($variant ( $($tcx.lift($variant_arg)?),* ))
                    }
                    $($output)*
                )
        )
    };

    (@Variants($this:expr, $tcx:expr)
     input( ($variant:path), $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumLiftImpl!(
            @Variants($this, $tcx)
                input($($input)*)
                output(
                    $variant => { Some($variant) }
                    $($output)*
                )
        )
    };
}

#[macro_export]
macro_rules! BraceStructTypeFoldableImpl {
    (impl<$($p:tt),*> TypeFoldable<$tcx:tt> for $s:path {
        $($field:ident),* $(,)?
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::fold::TypeFoldable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn super_fold_with<'gcx: $tcx, V: $crate::ty::fold::TypeFolder<'gcx, $tcx>>(
                &self,
                folder: &mut V,
            ) -> Result<Self, V::Error> {
                let $s { $($field,)* } = self;
                // FIXME(#58274): investigate parser bug
                Ok({ $s {
                    $($field: $crate::ty::fold::TypeFoldable::fold_with($field, folder)?),*
                }})
            }

            fn super_visit_with<V: $crate::ty::fold::TypeVisitor<$tcx>>(
                &self,
                visitor: &mut V,
            ) -> Result<(), V::Error> {
                let $s { $($field,)* } = self;
                $($crate::ty::fold::TypeFoldable::visit_with($field, visitor)?;)*;
                Ok(())
            }
        }
    };
}

#[macro_export]
macro_rules! TupleStructTypeFoldableImpl {
    (impl<$($p:tt),*> TypeFoldable<$tcx:tt> for $s:path {
        $($field:ident),* $(,)?
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::fold::TypeFoldable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn super_fold_with<'gcx: $tcx, V: $crate::ty::fold::TypeFolder<'gcx, $tcx>>(
                &self,
                folder: &mut V,
            ) -> Result<Self, V::Error> {
                let $s($($field,)*)= self;
                Ok($s(
                    $($crate::ty::fold::TypeFoldable::fold_with($field, folder)?),*
                ))
            }

            fn super_visit_with<V: $crate::ty::fold::TypeVisitor<$tcx>>(
                &self,
                visitor: &mut V,
            ) -> Result<(), V::Error> {
                let $s($($field,)*) = self;
                $($crate::ty::fold::TypeFoldable::visit_with($field, visitor)?;)*;
                Ok(())
            }
        }
    };
}

#[macro_export]
macro_rules! EnumTypeFoldableImpl {
    (impl<$($p:tt),*> TypeFoldable<$tcx:tt> for $s:path {
        $($variants:tt)*
    } $(where $($wc:tt)*)*) => {
        impl<$($p),*> $crate::ty::fold::TypeFoldable<$tcx> for $s
            $(where $($wc)*)*
        {
            fn super_fold_with<'gcx: $tcx, V: $crate::ty::fold::TypeFolder<'gcx, $tcx>>(
                &self,
                folder: &mut V,
            ) -> Result<Self, V::Error> {
                EnumTypeFoldableImpl!(@FoldVariants(self, folder) input($($variants)*) output())
            }

            fn super_visit_with<V: $crate::ty::fold::TypeVisitor<$tcx>>(
                &self,
                visitor: &mut V,
            ) -> Result<(), V::Error> {
                EnumTypeFoldableImpl!(@VisitVariants(self, visitor) input($($variants)*) output())
            }
        }
    };

    (@FoldVariants($this:expr, $folder:expr) input() output($($output:tt)*)) => {
        match $this {
            $($output)*
        }
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path) ( $($variant_arg:ident),* ) , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeFoldableImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant ( $($variant_arg),* ) => {
                        Ok($variant (
                            $($crate::ty::fold::TypeFoldable::fold_with($variant_arg, $folder)?),*
                        ))
                    }
                    $($output)*
                )
        )
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path) { $($variant_arg:ident),* $(,)? } , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeFoldableImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant { $($variant_arg),* } => {
                        // FIXME(#58274): investigate parser bug
                        Ok({$variant {
                            $($variant_arg: $crate::ty::fold::TypeFoldable::fold_with(
                                $variant_arg, $folder
                            )?),*
                        }})
                    }
                    $($output)*
                )
        )
    };

    (@FoldVariants($this:expr, $folder:expr)
     input( ($variant:path), $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeFoldableImpl!(
            @FoldVariants($this, $folder)
                input($($input)*)
                output(
                    $variant => { Ok($variant) }
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
        EnumTypeFoldableImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant ( $($variant_arg),* ) => {
                        $($crate::ty::fold::TypeFoldable::visit_with(
                            $variant_arg, $visitor
                        )?;)*;
                        Ok(())
                    }
                    $($output)*
                )
        )
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path) { $($variant_arg:ident),* $(,)? } , $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeFoldableImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant { $($variant_arg),* } => {
                        $($crate::ty::fold::TypeFoldable::visit_with(
                            $variant_arg, $visitor
                        )?;)*
                        Ok(())
                    }
                    $($output)*
                )
        )
    };

    (@VisitVariants($this:expr, $visitor:expr)
     input( ($variant:path), $($input:tt)*)
     output( $($output:tt)*) ) => {
        EnumTypeFoldableImpl!(
            @VisitVariants($this, $visitor)
                input($($input)*)
                output(
                    $variant => { Ok(()) }
                    $($output)*
                )
        )
    };
}
