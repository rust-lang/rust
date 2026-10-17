use super::{Attribute, DEPRECATED_CFG_ATTR, DEPRECATED_CLIPPY_CFG_ATTR, unnecessary_clippy_cfg};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::msrvs::{self, MsrvStack};
use clippy_utils::sym;
use rustc_ast::AttrStyle;
use rustc_errors::Applicability;
use rustc_lint::EarlyContext;

pub(super) fn check(cx: &EarlyContext<'_>, attr: &Attribute, msrv: &MsrvStack) {
    // check cfg_attr
    if attr.has_name(sym::cfg_attr)
        && let Some(items) = attr.meta_item_list()
        && items.len() == 2
        && let Some(feature_item) = items[0].meta_item()
    {
        // check for `rustfmt`
        if feature_item.has_name(sym::rustfmt)
            && msrv.meets(msrvs::TOOL_ATTRIBUTES)
            // check for `rustfmt_skip` and `rustfmt::skip`
            && let Some(skip_item) = &items[1].meta_item()
            && (skip_item.has_name(sym::rustfmt_skip)
                || skip_item
                    .path
                    .segments
                    .last()
                    .expect("empty path in attribute")
                    .ident
                    .name
                    == sym::skip)
            // Only lint outer attributes, because custom inner attributes are unstable
            // Tracking issue: https://github.com/rust-lang/rust/issues/54726
            && attr.style == AttrStyle::Outer
        {
            span_lint_and_sugg(
                cx,
                DEPRECATED_CFG_ATTR,
                attr.span,
                "`cfg_attr` is deprecated for rustfmt and got replaced by tool attributes",
                "use",
                "#[rustfmt::skip]".to_string(),
                Applicability::MachineApplicable,
            );
        } else {
            check_deprecated_cfg_recursively(cx, feature_item);
            if let Some(behind_cfg_attr) = items[1].meta_item() {
                unnecessary_clippy_cfg::check(cx, feature_item, behind_cfg_attr, attr);
            }
        }
    }
}

pub(super) fn check_clippy(cx: &EarlyContext<'_>, attr: &Attribute) {
    if attr.has_name(sym::cfg)
        && let Some(list) = attr.meta_item_list()
    {
        for item in list.iter().filter_map(|item| item.meta_item()) {
            check_deprecated_cfg_recursively(cx, item);
        }
    }
}

fn check_deprecated_cfg_recursively(cx: &EarlyContext<'_>, attr: &rustc_ast::MetaItem) {
    if let Some(ident) = attr.ident() {
        if matches!(ident.name, sym::any | sym::all | sym::not) {
            let Some(list) = attr.meta_item_list() else { return };
            for item in list.iter().filter_map(|item| item.meta_item()) {
                check_deprecated_cfg_recursively(cx, item);
            }
        } else {
            check_cargo_clippy_attr(cx, attr);
        }
    }
}

fn check_cargo_clippy_attr(cx: &EarlyContext<'_>, item: &rustc_ast::MetaItem) {
    if item.has_name(sym::feature) && item.value_str() == Some(sym::cargo_clippy) {
        span_lint_and_sugg(
            cx,
            DEPRECATED_CLIPPY_CFG_ATTR,
            item.span,
            "`feature = \"cargo-clippy\"` was replaced by `clippy`",
            "replace with",
            "clippy".to_string(),
            Applicability::MachineApplicable,
        );
    }
}
