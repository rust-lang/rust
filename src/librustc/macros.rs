// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
        $crate::session::bug_fmt(file!(), line!(), format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! span_bug {
    ($span:expr, $($message:tt)*) => ({
        $crate::session::span_bug_fmt(file!(), line!(), $span, format_args!($($message)*))
    })
}

#[macro_export]
macro_rules! __impl_stable_hash_field {
    (DECL IGNORED) => (_);
    (DECL $name:ident) => (ref $name);
    (USE IGNORED $ctx:expr, $hasher:expr) => ({});
    (USE $name:ident, $ctx:expr, $hasher:expr) => ($name.hash_stable($ctx, $hasher));
}

#[macro_export]
macro_rules! impl_stable_hash_for {
    (enum $enum_name:path { $( $variant:ident $( ( $($arg:ident),* ) )* ),* }) => {
        impl<'a, 'tcx> ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a, 'tcx>> for $enum_name {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a, 'tcx>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                use $enum_name::*;
                ::std::mem::discriminant(self).hash_stable(__ctx, __hasher);

                match *self {
                    $(
                        $variant $( ( $( __impl_stable_hash_field!(DECL $arg) ),* ) )* => {
                            $($( __impl_stable_hash_field!(USE $arg, __ctx, __hasher) );*)*
                        }
                    )*
                }
            }
        }
    };
    (struct $struct_name:path { $($field:ident),* }) => {
        impl<'a, 'tcx> ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a, 'tcx>> for $struct_name {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a, 'tcx>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                let $struct_name {
                    $(ref $field),*
                } = *self;

                $( $field.hash_stable(__ctx, __hasher));*
            }
        }
    };
    (tuple_struct $struct_name:path { $($field:ident),* }) => {
        impl<'a, 'tcx> ::rustc_data_structures::stable_hasher::HashStable<$crate::ich::StableHashingContext<'a, 'tcx>> for $struct_name {
            #[inline]
            fn hash_stable<W: ::rustc_data_structures::stable_hasher::StableHasherResult>(&self,
                                                  __ctx: &mut $crate::ich::StableHashingContext<'a, 'tcx>,
                                                  __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<W>) {
                let $struct_name (
                    $(ref $field),*
                ) = *self;

                $( $field.hash_stable(__ctx, __hasher));*
            }
        }
    };
}

#[macro_export]
macro_rules! impl_stable_hash_for_spanned {
    ($T:path) => (

        impl<'a, 'tcx> HashStable<StableHashingContext<'a, 'tcx>> for ::syntax::codemap::Spanned<$T>
        {
            #[inline]
            fn hash_stable<W: StableHasherResult>(&self,
                                                  hcx: &mut StableHashingContext<'a, 'tcx>,
                                                  hasher: &mut StableHasher<W>) {
                self.node.hash_stable(hcx, hasher);
                self.span.hash_stable(hcx, hasher);
            }
        }
    );
}

