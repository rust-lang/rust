use clippy_config::Conf;
use clippy_config::types::{
    SourceItemOrderingCategory, SourceItemOrderingModuleItemGroupings, SourceItemOrderingModuleItemKind,
    SourceItemOrderingTraitAssocItemKind, SourceItemOrderingTraitAssocItemKinds,
};
use clippy_utils::diagnostics::span_lint_and_note;
use rustc_hir::{
    AssocItemKind, FieldDef, HirId, ImplItemRef, IsAuto, Item, ItemKind, Mod, QPath, TraitItemRef, TyKind, UseKind,
    Variant, VariantData,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::impl_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    ///
    /// Confirms that items are sorted in source files as per configuration.
    ///
    /// ### Why restrict this?
    ///
    /// Keeping a consistent ordering throughout the codebase helps with working
    /// as a team, and possibly improves maintainability of the codebase. The
    /// idea is that by defining a consistent and enforceable rule for how
    /// source files are structured, less time will be wasted during reviews on
    /// a topic that is (under most circumstances) not relevant to the logic
    /// implemented in the code. Sometimes this will be referred to as
    /// "bikeshedding".
    ///
    /// ### Default Ordering and Configuration
    ///
    /// As there is no generally applicable rule, and each project may have
    /// different requirements, the lint can be configured with high
    /// granularity. The configuration is split into two stages:
    ///
    /// 1. Which item kinds that should have an internal order enforced.
    /// 2. Individual ordering rules per item kind.
    ///
    /// The item kinds that can be linted are:
    /// - Module (with customized groupings, alphabetical within)
    /// - Trait (with customized order of associated items, alphabetical within)
    /// - Enum, Impl, Struct (purely alphabetical)
    ///
    /// #### Module Item Order
    ///
    /// Due to the large variation of items within modules, the ordering can be
    /// configured on a very granular level. Item kinds can be grouped together
    /// arbitrarily, items within groups will be ordered alphabetically. The
    /// following table shows the default groupings:
    ///
    /// | Group              | Item Kinds           |
    /// |--------------------|----------------------|
    /// | `modules`          | "mod", "foreign_mod" |
    /// | `use`              | "use"                |
    /// | `macros`           | "macro"              |
    /// | `global_asm`       | "global_asm"         |
    /// | `UPPER_SNAKE_CASE` | "static", "const"    |
    /// | `PascalCase`       | "ty_alias", "opaque_ty", "enum", "struct", "union", "trait", "trait_alias", "impl" |
    /// | `lower_snake_case` | "fn"                 |
    ///
    /// All item kinds must be accounted for to create an enforceable linting
    /// rule set.
    ///
    /// ### Known Problems
    ///
    /// #### Performance Impact
    ///
    /// Keep in mind, that ordering source code alphabetically can lead to
    /// reduced performance in cases where the most commonly used enum variant
    /// isn't the first entry anymore, and similar optimizations that can reduce
    /// branch misses, cache locality and such. Either don't use this lint if
    /// that's relevant, or disable the lint in modules or items specifically
    /// where it matters. Other solutions can be to use profile guided
    /// optimization (PGO), post-link optimization (e.g. using BOLT for LLVM),
    /// or other advanced optimization methods. A good starting point to dig
    /// into optimization is [cargo-pgo][cargo-pgo].
    ///
    /// #### Lints on a Contains basis
    ///
    /// The lint can be disabled only on a "contains" basis, but not per element
    /// within a "container", e.g. the lint works per-module, per-struct,
    /// per-enum, etc. but not for "don't order this particular enum variant".
    ///
    /// #### Module documentation
    ///
    /// Module level rustdoc comments are not part of the resulting syntax tree
    /// and as such cannot be linted from within `check_mod`. Instead, the
    /// `rustdoc::missing_documentation` lint may be used.
    ///
    /// #### Module Tests
    ///
    /// This lint does not implement detection of module tests (or other feature
    /// dependent elements for that matter). To lint the location of mod tests,
    /// the lint `items_after_test_module` can be used instead.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// trait TraitUnordered {
    ///     const A: bool;
    ///     const C: bool;
    ///     const B: bool;
    ///
    ///     type SomeType;
    ///
    ///     fn a();
    ///     fn c();
    ///     fn b();
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// trait TraitOrdered {
    ///     const A: bool;
    ///     const B: bool;
    ///     const C: bool;
    ///
    ///     type SomeType;
    ///
    ///     fn a();
    ///     fn b();
    ///     fn c();
    /// }
    /// ```
    ///
    /// [cargo-pgo]: https://github.com/Kobzol/cargo-pgo/blob/main/README.md
    ///
    #[clippy::version = "1.82.0"]
    pub ARBITRARY_SOURCE_ITEM_ORDERING,
    restriction,
    "arbitrary source item ordering"
}

impl_lint_pass!(ArbitrarySourceItemOrdering => [ARBITRARY_SOURCE_ITEM_ORDERING]);

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)] // Bools are cached feature flags.
pub struct ArbitrarySourceItemOrdering {
    assoc_types_order: SourceItemOrderingTraitAssocItemKinds,
    enable_ordering_for_enum: bool,
    enable_ordering_for_impl: bool,
    enable_ordering_for_module: bool,
    enable_ordering_for_struct: bool,
    enable_ordering_for_trait: bool,
    module_item_order_groupings: SourceItemOrderingModuleItemGroupings,
}

impl ArbitrarySourceItemOrdering {
    pub fn new(conf: &'static Conf) -> Self {
        #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
        use SourceItemOrderingCategory::*;
        Self {
            assoc_types_order: conf.trait_assoc_item_kinds_order.clone(),
            enable_ordering_for_enum: conf.source_item_ordering.contains(&Enum),
            enable_ordering_for_impl: conf.source_item_ordering.contains(&Impl),
            enable_ordering_for_module: conf.source_item_ordering.contains(&Module),
            enable_ordering_for_struct: conf.source_item_ordering.contains(&Struct),
            enable_ordering_for_trait: conf.source_item_ordering.contains(&Trait),
            module_item_order_groupings: conf.module_item_order_groupings.clone(),
        }
    }

    /// Produces a linting warning for incorrectly ordered impl items.
    fn lint_impl_item<T: LintContext>(&self, cx: &T, item: &ImplItemRef, before_item: &ImplItemRef) {
        span_lint_and_note(
            cx,
            ARBITRARY_SOURCE_ITEM_ORDERING,
            item.span,
            format!(
                "incorrect ordering of impl items (defined order: {:?})",
                self.assoc_types_order
            ),
            Some(before_item.span),
            format!("should be placed before `{}`", before_item.ident.as_str(),),
        );
    }

    /// Produces a linting warning for incorrectly ordered item members.
    fn lint_member_name<T: LintContext>(
        cx: &T,
        ident: &rustc_span::symbol::Ident,
        before_ident: &rustc_span::symbol::Ident,
    ) {
        span_lint_and_note(
            cx,
            ARBITRARY_SOURCE_ITEM_ORDERING,
            ident.span,
            "incorrect ordering of items (must be alphabetically ordered)",
            Some(before_ident.span),
            format!("should be placed before `{}`", before_ident.as_str(),),
        );
    }

    fn lint_member_item<T: LintContext>(cx: &T, item: &Item<'_>, before_item: &Item<'_>) {
        let span = if item.ident.as_str().is_empty() {
            &item.span
        } else {
            &item.ident.span
        };

        let (before_span, note) = if before_item.ident.as_str().is_empty() {
            (
                &before_item.span,
                "should be placed before the following item".to_owned(),
            )
        } else {
            (
                &before_item.ident.span,
                format!("should be placed before `{}`", before_item.ident.as_str(),),
            )
        };

        // This catches false positives where generated code gets linted.
        if span == before_span {
            return;
        }

        span_lint_and_note(
            cx,
            ARBITRARY_SOURCE_ITEM_ORDERING,
            *span,
            "incorrect ordering of items (must be alphabetically ordered)",
            Some(*before_span),
            note,
        );
    }

    /// Produces a linting warning for incorrectly ordered trait items.
    fn lint_trait_item<T: LintContext>(&self, cx: &T, item: &TraitItemRef, before_item: &TraitItemRef) {
        span_lint_and_note(
            cx,
            ARBITRARY_SOURCE_ITEM_ORDERING,
            item.span,
            format!(
                "incorrect ordering of trait items (defined order: {:?})",
                self.assoc_types_order
            ),
            Some(before_item.span),
            format!("should be placed before `{}`", before_item.ident.as_str(),),
        );
    }
}

impl<'tcx> LateLintPass<'tcx> for ArbitrarySourceItemOrdering {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        match &item.kind {
            ItemKind::Enum(enum_def, _generics) if self.enable_ordering_for_enum => {
                let mut cur_v: Option<&Variant<'_>> = None;
                for variant in enum_def.variants {
                    if in_external_macro(cx.sess(), variant.span) {
                        continue;
                    }

                    if let Some(cur_v) = cur_v {
                        if cur_v.ident.name.as_str() > variant.ident.name.as_str() && cur_v.span != variant.span {
                            Self::lint_member_name(cx, &variant.ident, &cur_v.ident);
                        }
                    }
                    cur_v = Some(variant);
                }
            },
            ItemKind::Struct(VariantData::Struct { fields, .. }, _generics) if self.enable_ordering_for_struct => {
                let mut cur_f: Option<&FieldDef<'_>> = None;
                for field in *fields {
                    if in_external_macro(cx.sess(), field.span) {
                        continue;
                    }

                    if let Some(cur_f) = cur_f {
                        if cur_f.ident.name.as_str() > field.ident.name.as_str() && cur_f.span != field.span {
                            Self::lint_member_name(cx, &field.ident, &cur_f.ident);
                        }
                    }
                    cur_f = Some(field);
                }
            },
            ItemKind::Trait(is_auto, _safety, _generics, _generic_bounds, item_ref)
                if self.enable_ordering_for_trait && *is_auto == IsAuto::No =>
            {
                let mut cur_t: Option<&TraitItemRef> = None;

                for item in *item_ref {
                    if in_external_macro(cx.sess(), item.span) {
                        continue;
                    }

                    if let Some(cur_t) = cur_t {
                        let cur_t_kind = convert_assoc_item_kind(cur_t.kind);
                        let cur_t_kind_index = self.assoc_types_order.index_of(&cur_t_kind);
                        let item_kind = convert_assoc_item_kind(item.kind);
                        let item_kind_index = self.assoc_types_order.index_of(&item_kind);

                        if cur_t_kind == item_kind && cur_t.ident.name.as_str() > item.ident.name.as_str() {
                            Self::lint_member_name(cx, &item.ident, &cur_t.ident);
                        } else if cur_t_kind_index > item_kind_index {
                            self.lint_trait_item(cx, item, cur_t);
                        }
                    }
                    cur_t = Some(item);
                }
            },
            ItemKind::Impl(trait_impl) if self.enable_ordering_for_impl => {
                let mut cur_t: Option<&ImplItemRef> = None;

                for item in trait_impl.items {
                    if in_external_macro(cx.sess(), item.span) {
                        continue;
                    }

                    if let Some(cur_t) = cur_t {
                        let cur_t_kind = convert_assoc_item_kind(cur_t.kind);
                        let cur_t_kind_index = self.assoc_types_order.index_of(&cur_t_kind);
                        let item_kind = convert_assoc_item_kind(item.kind);
                        let item_kind_index = self.assoc_types_order.index_of(&item_kind);

                        if cur_t_kind == item_kind && cur_t.ident.name.as_str() > item.ident.name.as_str() {
                            Self::lint_member_name(cx, &item.ident, &cur_t.ident);
                        } else if cur_t_kind_index > item_kind_index {
                            self.lint_impl_item(cx, item, cur_t);
                        }
                    }
                    cur_t = Some(item);
                }
            },
            _ => {}, // Catch-all for `ItemKinds` that don't have fields.
        }
    }

    fn check_mod(&mut self, cx: &LateContext<'tcx>, module: &'tcx Mod<'tcx>, _: HirId) {
        struct CurItem<'a> {
            item: &'a Item<'a>,
            order: usize,
            name: String,
        }
        let mut cur_t: Option<CurItem<'_>> = None;

        if !self.enable_ordering_for_module {
            return;
        }

        let items = module.item_ids.iter().map(|&id| cx.tcx.hir().item(id));

        // Iterates over the items within a module.
        //
        // As of 2023-05-09, the Rust compiler will hold the entries in the same
        // order as they appear in the source code, which is convenient for us,
        // as no sorting by source map/line of code has to be applied.
        //
        for item in items {
            if in_external_macro(cx.sess(), item.span) {
                continue;
            }

            // The following exceptions (skipping with `continue;`) may not be
            // complete, edge cases have not been explored further than what
            // appears in the existing code base.
            if item.ident.name == rustc_span::symbol::kw::Empty {
                if let ItemKind::Impl(_) = item.kind {
                    // Sorting trait impls for unnamed types makes no sense.
                    if get_item_name(item).is_empty() {
                        continue;
                    }
                } else if let ItemKind::ForeignMod { .. } = item.kind {
                    continue;
                } else if let ItemKind::GlobalAsm(_) = item.kind {
                    continue;
                } else if let ItemKind::Use(path, use_kind) = item.kind {
                    if path.segments.is_empty() {
                        // Use statements that contain braces get caught here.
                        // They will still be linted internally.
                        continue;
                    } else if path.segments.len() >= 2
                        && (path.segments[0].ident.name == rustc_span::sym::std
                            || path.segments[0].ident.name == rustc_span::sym::core)
                        && path.segments[1].ident.name == rustc_span::sym::prelude
                    {
                        // Filters the autogenerated prelude use statement.
                        // e.g. `use std::prelude::rustc_2021`
                    } else if use_kind == UseKind::Glob {
                        // Filters glob kinds of uses.
                        // e.g. `use std::sync::*`
                    } else {
                        // This can be used for debugging.
                        // println!("Unknown autogenerated use statement: {:?}", item);
                    }
                    continue;
                }
            }

            if item.ident.name.as_str().starts_with('_') {
                // Filters out unnamed macro-like impls for various derives,
                // e.g. serde::Serialize or num_derive::FromPrimitive.
                continue;
            }

            if item.ident.name == rustc_span::sym::std && item.span.is_dummy() {
                if let ItemKind::ExternCrate(None) = item.kind {
                    // Filters the auto-included Rust standard library.
                    continue;
                }
                println!("Unknown item: {item:?}");
            }

            let item_kind = convert_module_item_kind(&item.kind);
            let module_level_order = self
                .module_item_order_groupings
                .module_level_order_of(&item_kind)
                .unwrap_or_default();

            if let Some(cur_t) = cur_t.as_ref() {
                use std::cmp::Ordering; // Better legibility.
                match module_level_order.cmp(&cur_t.order) {
                    Ordering::Less => {
                        Self::lint_member_item(cx, item, cur_t.item);
                    },
                    Ordering::Equal if item_kind == SourceItemOrderingModuleItemKind::Use => {
                        // Skip ordering use statements, as these should be ordered by rustfmt.
                    },
                    Ordering::Equal if cur_t.name > get_item_name(item) => {
                        Self::lint_member_item(cx, item, cur_t.item);
                    },
                    Ordering::Equal | Ordering::Greater => {
                        // Nothing to do in this case, they're already in the right order.
                    },
                }
            }

            // Makes a note of the current item for comparison with the next.
            cur_t = Some(CurItem {
                order: module_level_order,
                item,
                name: get_item_name(item),
            });
        }
    }
}

/// Converts a [`rustc_hir::AssocItemKind`] to a
/// [`SourceItemOrderingTraitAssocItemKind`].
///
/// This is implemented here because `rustc_hir` is not a dependency of
/// `clippy_config`.
fn convert_assoc_item_kind(value: AssocItemKind) -> SourceItemOrderingTraitAssocItemKind {
    #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
    use SourceItemOrderingTraitAssocItemKind::*;
    match value {
        AssocItemKind::Const { .. } => Const,
        AssocItemKind::Type { .. } => Type,
        AssocItemKind::Fn { .. } => Fn,
    }
}

/// Converts a [`rustc_hir::ItemKind`] to a
/// [`SourceItemOrderingModuleItemKind`].
///
/// This is implemented here because `rustc_hir` is not a dependency of
/// `clippy_config`.
fn convert_module_item_kind(value: &ItemKind<'_>) -> SourceItemOrderingModuleItemKind {
    #[allow(clippy::enum_glob_use)] // Very local glob use for legibility.
    use SourceItemOrderingModuleItemKind::*;
    match value {
        ItemKind::ExternCrate(..) => ExternCrate,
        ItemKind::Use(..) => Use,
        ItemKind::Static(..) => Static,
        ItemKind::Const(..) => Const,
        ItemKind::Fn(..) => Fn,
        ItemKind::Macro(..) => Macro,
        ItemKind::Mod(..) => Mod,
        ItemKind::ForeignMod { .. } => ForeignMod,
        ItemKind::GlobalAsm(..) => GlobalAsm,
        ItemKind::TyAlias(..) => TyAlias,
        ItemKind::Enum(..) => Enum,
        ItemKind::Struct(..) => Struct,
        ItemKind::Union(..) => Union,
        ItemKind::Trait(..) => Trait,
        ItemKind::TraitAlias(..) => TraitAlias,
        ItemKind::Impl(..) => Impl,
    }
}

/// Gets the item name for sorting purposes, which in the general case is
/// `item.ident.name`.
///
/// For trait impls, the name used for sorting will be the written path of
/// `item.self_ty` plus the written path of `item.of_trait`, joined with
/// exclamation marks. Exclamation marks are used because they are the first
/// printable ASCII character.
///
/// Trait impls generated using a derive-macro will have their path rewritten,
/// such that for example `Default` is `$crate::default::Default`, and
/// `std::clone::Clone` is `$crate::clone::Clone`. This behaviour is described
/// further in the [Rust Reference, Paths Chapter][rust_ref].
///
/// [rust_ref]: https://doc.rust-lang.org/reference/paths.html#crate-1
fn get_item_name(item: &Item<'_>) -> String {
    match item.kind {
        ItemKind::Impl(im) => {
            if let TyKind::Path(path) = im.self_ty.kind {
                match path {
                    QPath::Resolved(_, path) => {
                        let segs = path.segments.iter();
                        let mut segs: Vec<String> = segs.map(|s| s.ident.name.as_str().to_owned()).collect();

                        if let Some(of_trait) = im.of_trait {
                            let mut trait_segs: Vec<String> = of_trait
                                .path
                                .segments
                                .iter()
                                .map(|s| s.ident.name.as_str().to_owned())
                                .collect();
                            segs.append(&mut trait_segs);
                        }

                        segs.push(String::new());
                        segs.join("!!")
                    },
                    QPath::TypeRelative(_, _path_seg) => {
                        // This case doesn't exist in the clippy tests codebase.
                        String::new()
                    },
                    QPath::LangItem(_, _) => String::new(),
                }
            } else {
                // Impls for anything that isn't a named type can be skipped.
                String::new()
            }
        },
        _ => item.ident.name.as_str().to_owned(),
    }
}
