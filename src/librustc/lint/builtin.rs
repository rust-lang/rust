// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Some lints that are built in to the compiler.
//!
//! These are the built-in lints that are emitted direct in the main
//! compiler code, rather than using their own custom pass. Those
//! lints are all available in `rustc_lint::builtin`.

use lint::{LintPass, LateLintPass, LintArray};

declare_lint! {
    pub CONST_ERR,
    Warn,
    "constant evaluation detected erroneous expression"
}

declare_lint! {
    pub UNUSED_IMPORTS,
    Warn,
    "imports that are never used"
}

declare_lint! {
    pub UNUSED_EXTERN_CRATES,
    Allow,
    "extern crates that are never used"
}

declare_lint! {
    pub UNUSED_QUALIFICATIONS,
    Allow,
    "detects unnecessarily qualified names"
}

declare_lint! {
    pub UNKNOWN_LINTS,
    Warn,
    "unrecognized lint attribute"
}

declare_lint! {
    pub UNUSED_VARIABLES,
    Warn,
    "detect variables which are not used in any way"
}

declare_lint! {
    pub UNUSED_ASSIGNMENTS,
    Warn,
    "detect assignments that will never be read"
}

declare_lint! {
    pub DEAD_CODE,
    Warn,
    "detect unused, unexported items"
}

declare_lint! {
    pub UNREACHABLE_CODE,
    Warn,
    "detects unreachable code paths"
}

declare_lint! {
    pub WARNINGS,
    Warn,
    "mass-change the level for lints which produce warnings"
}

declare_lint! {
    pub UNUSED_FEATURES,
    Warn,
    "unused or unknown features found in crate-level #[feature] directives"
}

declare_lint! {
    pub STABLE_FEATURES,
    Warn,
    "stable features found in #[feature] directive"
}

declare_lint! {
    pub UNKNOWN_CRATE_TYPES,
    Deny,
    "unknown crate type found in #[crate_type] directive"
}

declare_lint! {
    pub FAT_PTR_TRANSMUTES,
    Allow,
    "detects transmutes of fat pointers"
}

declare_lint! {
    pub TRIVIAL_CASTS,
    Allow,
    "detects trivial casts which could be removed"
}

declare_lint! {
    pub TRIVIAL_NUMERIC_CASTS,
    Allow,
    "detects trivial casts of numeric types which could be removed"
}

declare_lint! {
    pub PRIVATE_IN_PUBLIC,
    Deny,
    "detect private items in public interfaces not caught by the old implementation"
}

declare_lint! {
    pub INACCESSIBLE_EXTERN_CRATE,
    Warn,
    "use of inaccessible extern crate erroneously allowed"
}

declare_lint! {
    pub INVALID_TYPE_PARAM_DEFAULT,
    Warn,
    "type parameter default erroneously allowed in invalid location"
}

declare_lint! {
    pub ILLEGAL_FLOATING_POINT_CONSTANT_PATTERN,
    Warn,
    "floating-point constants cannot be used in patterns"
}

declare_lint! {
    pub ILLEGAL_STRUCT_OR_ENUM_CONSTANT_PATTERN,
    Warn,
    "constants of struct or enum type can only be used in a pattern if \
     the struct or enum has `#[derive(PartialEq, Eq)]`"
}

declare_lint! {
    pub MATCH_OF_UNIT_VARIANT_VIA_PAREN_DOTDOT,
    Deny,
    "unit struct or enum variant erroneously allowed to match via path::ident(..)"
}

declare_lint! {
    pub RAW_POINTER_DERIVE,
    Warn,
    "uses of #[derive] with raw pointers are rarely correct"
}

declare_lint! {
    pub TRANSMUTE_FROM_FN_ITEM_TYPES,
    Deny,
    "transmute from function item type to pointer-sized type erroneously allowed"
}

declare_lint! {
    pub HR_LIFETIME_IN_ASSOC_TYPE,
    Warn,
    "binding for associated type references higher-ranked lifetime \
     that does not appear in the trait input types"
}

declare_lint! {
    pub OVERLAPPING_INHERENT_IMPLS,
    Warn,
    "two overlapping inherent impls define an item with the same name were erroneously allowed"
}

declare_lint! {
    pub RENAMED_AND_REMOVED_LINTS,
    Warn,
    "lints that have been renamed or removed"
}

declare_lint! {
    pub SUPER_OR_SELF_IN_GLOBAL_PATH,
    Warn,
    "detects super or self keywords at the beginning of global path"
}

declare_lint! {
    pub LIFETIME_UNDERSCORE,
    Warn,
    "lifetimes or labels named `'_` were erroneously allowed"
}

/// Does nothing as a lint pass, but registers some `Lint`s
/// which are used by other parts of the compiler.
#[derive(Copy, Clone)]
pub struct HardwiredLints;

impl LintPass for HardwiredLints {
    fn get_lints(&self) -> LintArray {
        lint_array!(
            UNUSED_IMPORTS,
            UNUSED_EXTERN_CRATES,
            UNUSED_QUALIFICATIONS,
            UNKNOWN_LINTS,
            UNUSED_VARIABLES,
            UNUSED_ASSIGNMENTS,
            DEAD_CODE,
            UNREACHABLE_CODE,
            WARNINGS,
            UNUSED_FEATURES,
            STABLE_FEATURES,
            UNKNOWN_CRATE_TYPES,
            FAT_PTR_TRANSMUTES,
            TRIVIAL_CASTS,
            TRIVIAL_NUMERIC_CASTS,
            PRIVATE_IN_PUBLIC,
            INACCESSIBLE_EXTERN_CRATE,
            INVALID_TYPE_PARAM_DEFAULT,
            ILLEGAL_FLOATING_POINT_CONSTANT_PATTERN,
            ILLEGAL_STRUCT_OR_ENUM_CONSTANT_PATTERN,
            MATCH_OF_UNIT_VARIANT_VIA_PAREN_DOTDOT,
            CONST_ERR,
            RAW_POINTER_DERIVE,
            TRANSMUTE_FROM_FN_ITEM_TYPES,
            OVERLAPPING_INHERENT_IMPLS,
            RENAMED_AND_REMOVED_LINTS,
            SUPER_OR_SELF_IN_GLOBAL_PATH,
            HR_LIFETIME_IN_ASSOC_TYPE,
            LIFETIME_UNDERSCORE
        )
    }
}

impl LateLintPass for HardwiredLints {}
