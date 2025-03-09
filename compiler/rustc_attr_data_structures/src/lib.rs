// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

mod attributes;
mod stability;
mod version;

use std::num::NonZero;

pub use attributes::*;
use rustc_abi::Align;
use rustc_ast::token::CommentKind;
use rustc_ast::{AttrStyle, IntTy, UintTy};
use rustc_ast_pretty::pp::Printer;
use rustc_span::hygiene::Transparency;
use rustc_span::{Span, Symbol};
pub use stability::*;
use thin_vec::ThinVec;
pub use version::*;

/// Requirements for a `StableHashingContext` to be used in this crate.
/// This is a hack to allow using the `HashStable_Generic` derive macro
/// instead of implementing everything in `rustc_middle`.
pub trait HashStableContext: rustc_ast::HashStableContext + rustc_abi::HashStableContext {}

/// This trait is used to print attributes in `rustc_hir_pretty`.
///
/// For structs and enums it can be derived using [`rustc_macros::PrintAttribute`].
/// The output will look a lot like a `Debug` implementation, but fields of several types
/// like [`Span`]s and empty tuples, are gracefully skipped so they don't clutter the
/// representation much.
pub trait PrintAttribute {
    fn print_something(&self) -> bool;
    fn print_attribute(&self, p: &mut Printer);
}

impl<T: PrintAttribute> PrintAttribute for &T {
    fn print_something(&self) -> bool {
        T::print_something(self)
    }

    fn print_attribute(&self, p: &mut Printer) {
        T::print_attribute(self, p)
    }
}
impl<T: PrintAttribute> PrintAttribute for Option<T> {
    fn print_something(&self) -> bool {
        self.as_ref().is_some_and(|x| x.print_something())
    }
    fn print_attribute(&self, p: &mut Printer) {
        if let Some(i) = self {
            T::print_attribute(i, p)
        }
    }
}
impl<T: PrintAttribute> PrintAttribute for ThinVec<T> {
    fn print_something(&self) -> bool {
        self.is_empty() || self[0].print_something()
    }
    fn print_attribute(&self, p: &mut Printer) {
        let mut last_printed = false;
        p.word("[");
        for i in self {
            if last_printed {
                p.word_space(",");
            }
            i.print_attribute(p);
            last_printed = i.print_something();
        }
        p.word("]");
    }
}
macro_rules! print_skip {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn print_something(&self) -> bool { false }
            fn print_attribute(&self, _: &mut Printer) { }
        })*
    };
}

macro_rules! print_disp {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn print_something(&self) -> bool { true }
            fn print_attribute(&self, p: &mut Printer) {
                p.word(format!("{}", self));
            }
        }
    )*};
}
macro_rules! print_debug {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn print_something(&self) -> bool { true }
            fn print_attribute(&self, p: &mut Printer) {
                p.word(format!("{:?}", self));
            }
        }
    )*};
}

macro_rules! print_tup {
    (num_print_something $($ts: ident)*) => { 0 $(+ $ts.print_something() as usize)* };
    () => {};
    ($t: ident $($ts: ident)*) => {
        #[allow(non_snake_case, unused)]
        impl<$t: PrintAttribute, $($ts: PrintAttribute),*> PrintAttribute for ($t, $($ts),*) {
            fn print_something(&self) -> bool {
                let ($t, $($ts),*) = self;
                print_tup!(num_print_something $t $($ts)*) != 0
            }

            fn print_attribute(&self, p: &mut Printer) {
                let ($t, $($ts),*) = self;
                let parens = print_tup!(num_print_something $t $($ts)*) > 1;
                if parens {
                    p.word("(");
                }

                let mut printed_anything = $t.print_something();

                $t.print_attribute(p);

                $(
                    if printed_anything && $ts.print_something() {
                        p.word_space(",");
                        printed_anything = true;
                    }
                    $ts.print_attribute(p);
                )*

                if parens {
                    p.word(")");
                }
            }
        }

        print_tup!($($ts)*);
    };
}

print_tup!(A B C D E F G H);
print_skip!(Span, ());
print_disp!(Symbol, u16, bool, NonZero<u32>);
print_debug!(UintTy, IntTy, Align, AttrStyle, CommentKind, Transparency);
