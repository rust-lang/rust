// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

mod attributes;
mod stability;
mod version;

pub mod lints;

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
    /// Whether or not this will render as something meaningful, or if it's skipped
    /// (which will force the containing struct to also skip printing a comma
    /// and the field name).
    fn should_render(&self) -> bool;

    fn print_attribute(&self, p: &mut Printer);
}

impl<T: PrintAttribute> PrintAttribute for &T {
    fn should_render(&self) -> bool {
        T::should_render(self)
    }

    fn print_attribute(&self, p: &mut Printer) {
        T::print_attribute(self, p)
    }
}
impl<T: PrintAttribute> PrintAttribute for Option<T> {
    fn should_render(&self) -> bool {
        self.as_ref().is_some_and(|x| x.should_render())
    }

    fn print_attribute(&self, p: &mut Printer) {
        if let Some(i) = self {
            T::print_attribute(i, p)
        }
    }
}
impl<T: PrintAttribute> PrintAttribute for ThinVec<T> {
    fn should_render(&self) -> bool {
        self.is_empty() || self[0].should_render()
    }

    fn print_attribute(&self, p: &mut Printer) {
        let mut last_printed = false;
        p.word("[");
        for i in self {
            if last_printed {
                p.word_space(",");
            }
            i.print_attribute(p);
            last_printed = i.should_render();
        }
        p.word("]");
    }
}
macro_rules! print_skip {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn should_render(&self) -> bool { false }
            fn print_attribute(&self, _: &mut Printer) { }
        })*
    };
}

macro_rules! print_disp {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn should_render(&self) -> bool { true }
            fn print_attribute(&self, p: &mut Printer) {
                p.word(format!("{}", self));
            }
        }
    )*};
}
macro_rules! print_debug {
    ($($t: ty),* $(,)?) => {$(
        impl PrintAttribute for $t {
            fn should_render(&self) -> bool { true }
            fn print_attribute(&self, p: &mut Printer) {
                p.word(format!("{:?}", self));
            }
        }
    )*};
}

macro_rules! print_tup {
    (num_should_render $($ts: ident)*) => { 0 $(+ $ts.should_render() as usize)* };
    () => {};
    ($t: ident $($ts: ident)*) => {
        #[allow(non_snake_case, unused)]
        impl<$t: PrintAttribute, $($ts: PrintAttribute),*> PrintAttribute for ($t, $($ts),*) {
            fn should_render(&self) -> bool {
                let ($t, $($ts),*) = self;
                print_tup!(num_should_render $t $($ts)*) != 0
            }

            fn print_attribute(&self, p: &mut Printer) {
                let ($t, $($ts),*) = self;
                let parens = print_tup!(num_should_render $t $($ts)*) > 1;
                if parens {
                    p.popen();
                }

                let mut printed_anything = $t.should_render();

                $t.print_attribute(p);

                $(
                    if $ts.should_render() {
                        if printed_anything {
                            p.word_space(",");
                        }
                        printed_anything = true;
                    }
                    $ts.print_attribute(p);
                )*

                if parens {
                    p.pclose();
                }
            }
        }

        print_tup!($($ts)*);
    };
}

print_tup!(A B C D E F G H);
print_skip!(Span, ());
print_disp!(u16, bool, NonZero<u32>);
print_debug!(Symbol, UintTy, IntTy, Align, AttrStyle, CommentKind, Transparency);

/// Finds attributes in sequences of attributes by pattern matching.
///
/// A little like `matches` but for attributes.
///
/// ```rust,ignore (illustrative)
/// // finds the repr attribute
/// if let Some(r) = find_attr!(attrs, AttributeKind::Repr(r) => r) {
///
/// }
///
/// // checks if one has matched
/// if find_attr!(attrs, AttributeKind::Repr(_)) {
///
/// }
/// ```
///
/// Often this requires you to first end up with a list of attributes.
/// A common way to get those is through `tcx.get_all_attrs(did)`
#[macro_export]
macro_rules! find_attr {
    ($attributes_list: expr, $pattern: pat $(if $guard: expr)?) => {{
        $crate::find_attr!($attributes_list, $pattern $(if $guard)? => ()).is_some()
    }};

    ($attributes_list: expr, $pattern: pat $(if $guard: expr)? => $e: expr) => {{
        'done: {
            for i in $attributes_list {
                let i: &rustc_hir::Attribute = i;
                match i {
                    rustc_hir::Attribute::Parsed($pattern) $(if $guard)? => {
                        break 'done Some($e);
                    }
                    _ => {}
                }
            }

            None
        }
    }};
}
