//! Removed features library for testing `#[unstable_removed]`.

#![allow(unused)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(missing_debug_implementations)]

/// ============================================================
/// 1. Removed const item
#[unstable_removed(
    feature = "removed_const_item",
    reason = "testing removed const",
    issue = "123456",
    since = "1.89.0"
)]
pub const REMOVED_CONST: u32 = 42;

#[unstable_removed(
    feature = "removed_const_item",
    reason = "testing removed const",
    issue = "123456",
    since = "1.89.0"
)]
pub mod const_removed {
    #[unstable_removed(
        feature = "removed_const_item",
        reason = "testing removed const",
        issue = "123456",
        since = "1.89.0"
    )]
    pub fn use_removed_const() -> u32 {
        super::REMOVED_CONST
    }
}

/// ============================================================
/// 2. Removed trait / default items
#[unstable_removed(
    feature = "removed_trait_fn",
    reason = "testing removed trait fn",
    issue = "123456",
    since = "1.89.0"
)]
pub trait TraitRemoved {
    #[unstable_removed(
        feature = "removed_trait_fn",
        reason = "testing removed trait fn",
        issue = "123456",
        since = "1.89.0"
    )]
    fn removed_fn() -> u32 {
        42
    }

    #[unstable_removed(
        feature = "removed_trait_fn",
        reason = "testing removed trait fn",
        issue = "123456",
        since = "1.89.0"
    )]
    const REMOVED_TRAIT_CONST: u32 = 1;
}

#[unstable_removed(
    feature = "removed_trait_impl",
    reason = "testing removed trait impl",
    issue = "123456",
    since = "1.89.0"
)]
pub struct Impl;

#[unstable_removed(
    feature = "removed_trait_impl",
    reason = "testing removed trait impl",
    issue = "123456",
    since = "1.89.0"
)]
impl TraitRemoved for Impl {
    fn removed_fn() -> u32 {
        0
    }
    const REMOVED_TRAIT_CONST: u32 = 0;
}

/// ============================================================
/// 3. Removed macro
#[unstable_removed(
    feature = "removed_macro_item",
    reason = "testing removed macro",
    issue = "123456",
    since = "1.89.0"
)]
#[macro_export]
macro_rules! removed_macro {
    () => {
        const REMOVED_MACRO_CONST: u32 = 0;
    };
}

/// ============================================================
/// 4. Removed function for missing field tests
#[unstable_removed(feature = "removed_no_reason", since = "1.89.0", issue = "123456")]
pub fn removed_no_reason() {}
