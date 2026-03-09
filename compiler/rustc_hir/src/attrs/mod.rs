//! Data structures for representing parsed attributes in the Rust compiler.
//! Formerly `rustc_attr_data_structures`.
//!
//! For detailed documentation about attribute processing,
//! see [rustc_attr_parsing](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_attr_parsing/index.html).

pub use data_structures::*;
pub use encode_cross_crate::EncodeCrossCrate;
pub use pretty_printing::PrintAttribute;

mod data_structures;
pub mod diagnostic;
mod encode_cross_crate;
mod pretty_printing;

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
/// Often these are available through the `tcx`.
///
/// As a convenience, this macro can do that for you!
///
/// Instead of providing an attribute list, provide the `tcx` and a `DefId`.
///
/// ```rust,ignore (illustrative)
/// find_attr!(tcx, def_id, <pattern>)
/// ```
///
/// Another common case is finding attributes applied to the root of the current crate.
/// For that, use the shortcut:
///
/// ```rust, ignore (illustrative)
/// find_attr!(tcx, crate, <pattern>)
/// ```
#[macro_export]
macro_rules! find_attr {
    ($tcx: expr, crate, $pattern: pat $(if $guard: expr)?) => {
        $crate::find_attr!($tcx, crate, $pattern $(if $guard)? => ()).is_some()
    };
    ($tcx: expr, crate, $pattern: pat $(if $guard: expr)? => $e: expr) => {
        $crate::find_attr!($tcx.hir_krate_attrs(), $pattern $(if $guard)? => $e)
    };

    ($tcx: expr, $def_id: expr, $pattern: pat $(if $guard: expr)?) => {
        $crate::find_attr!($tcx, $def_id, $pattern $(if $guard)? => ()).is_some()
    };
    ($tcx: expr, $def_id: expr, $pattern: pat $(if $guard: expr)? => $e: expr) => {{
        #[allow(deprecated)] {
            $crate::find_attr!($tcx.get_all_attrs($def_id), $pattern $(if $guard)? => $e)
        }
    }};


    ($attributes_list: expr, $pattern: pat $(if $guard: expr)?) => {{
        $crate::find_attr!($attributes_list, $pattern $(if $guard)? => ()).is_some()
    }};

    ($attributes_list: expr, $pattern: pat $(if $guard: expr)? => $e: expr) => {{
        'done: {
            for i in $attributes_list {
                #[allow(unused_imports)]
                use rustc_hir::attrs::AttributeKind::*;
                let i: &rustc_hir::Attribute = i;
                match i {
                    rustc_hir::Attribute::Parsed($pattern) $(if $guard)? => {
                        break 'done Some($e);
                    }
                    rustc_hir::Attribute::Unparsed(..) => {}
                    // In lint emitting, there's a specific exception for this warning.
                    // It's not usually emitted from inside macros from other crates
                    // (see https://github.com/rust-lang/rust/issues/110613)
                    // But this one is!
                    #[deny(unreachable_patterns)]
                    _ => {}
                }
            }

            None
        }
    }};
}
