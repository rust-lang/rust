use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_and_help, span_lint_hir};
use clippy_utils::is_bool;
use clippy_utils::macros::span_is_local;
use clippy_utils::source::is_present_in_source;
use clippy_utils::str_utils::{camel_case_split, count_match_end, count_match_start, to_camel_case, to_snake_case};
use rustc_data_structures::fx::FxHashSet;
use rustc_hir::{EnumDef, FieldDef, Item, ItemKind, OwnerId, QPath, TyKind, Variant, VariantData};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Detects enumeration variants that are prefixed or suffixed
    /// by the same characters.
    ///
    /// ### Why is this bad?
    /// Enumeration variant names should specify their variant,
    /// not repeat the enumeration name.
    ///
    /// ### Limitations
    /// Characters with no casing will be considered when comparing prefixes/suffixes
    /// This applies to numbers and non-ascii characters without casing
    /// e.g. `Foo1` and `Foo2` is considered to have different prefixes
    /// (the prefixes are `Foo1` and `Foo2` respectively), as also `Bar螃`, `Bar蟹`
    ///
    /// ### Example
    /// ```no_run
    /// enum Cake {
    ///     BlackForestCake,
    ///     HummingbirdCake,
    ///     BattenbergCake,
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// enum Cake {
    ///     BlackForest,
    ///     Hummingbird,
    ///     Battenberg,
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub ENUM_VARIANT_NAMES,
    style,
    "enums where all variants share a prefix/postfix"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects public item names that are prefixed or suffixed by the
    /// containing public module's name.
    ///
    /// ### Why is this bad?
    /// It requires the user to type the module name twice in each usage,
    /// especially if they choose to import the module rather than its contents.
    ///
    /// Lack of such repetition is also the style used in the Rust standard library;
    /// e.g. `io::Error` and `fmt::Error` rather than `io::IoError` and `fmt::FmtError`;
    /// and `array::from_ref` rather than `array::array_from_ref`.
    ///
    /// ### Known issues
    /// Glob re-exports are ignored; e.g. this will not warn even though it should:
    ///
    /// ```no_run
    /// pub mod foo {
    ///     mod iteration {
    ///         pub struct FooIter {}
    ///     }
    ///     pub use iteration::*; // creates the path `foo::FooIter`
    /// }
    /// ```
    ///
    /// ### Example
    /// ```no_run
    /// mod cake {
    ///     struct BlackForestCake;
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// mod cake {
    ///     struct BlackForest;
    /// }
    /// ```
    #[clippy::version = "1.33.0"]
    pub MODULE_NAME_REPETITIONS,
    restriction,
    "type names prefixed/postfixed with their containing module's name"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for modules that have the same name as their
    /// parent module
    ///
    /// ### Why is this bad?
    /// A typical beginner mistake is to have `mod foo;` and
    /// again `mod foo { ..
    /// }` in `foo.rs`.
    /// The expectation is that items inside the inner `mod foo { .. }` are then
    /// available
    /// through `foo::x`, but they are only available through
    /// `foo::foo::x`.
    /// If this is done on purpose, it would be better to choose a more
    /// representative module name.
    ///
    /// ### Example
    /// ```ignore
    /// // lib.rs
    /// mod foo;
    /// // foo.rs
    /// mod foo {
    ///     ...
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MODULE_INCEPTION,
    style,
    "modules that have the same name as their parent module"
}
declare_clippy_lint! {
    /// ### What it does
    /// Detects struct fields that are prefixed or suffixed
    /// by the same characters or the name of the struct itself.
    ///
    /// ### Why is this bad?
    /// Information common to all struct fields is better represented in the struct name.
    ///
    /// ### Limitations
    /// Characters with no casing will be considered when comparing prefixes/suffixes
    /// This applies to numbers and non-ascii characters without casing
    /// e.g. `foo1` and `foo2` is considered to have different prefixes
    /// (the prefixes are `foo1` and `foo2` respectively), as also `bar螃`, `bar蟹`
    ///
    /// ### Example
    /// ```no_run
    /// struct Cake {
    ///     cake_sugar: u8,
    ///     cake_flour: u8,
    ///     cake_eggs: u8
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct Cake {
    ///     sugar: u8,
    ///     flour: u8,
    ///     eggs: u8
    /// }
    /// ```
    #[clippy::version = "1.75.0"]
    pub STRUCT_FIELD_NAMES,
    pedantic,
    "structs where all fields share a prefix/postfix or contain the name of the struct"
}

pub struct ItemNameRepetitions {
    modules: Vec<(Symbol, String, OwnerId)>,
    enum_threshold: u64,
    struct_threshold: u64,
    avoid_breaking_exported_api: bool,
    allow_exact_repetitions: bool,
    allow_private_module_inception: bool,
    allowed_prefixes: FxHashSet<String>,
}

impl ItemNameRepetitions {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            modules: Vec::new(),
            enum_threshold: conf.enum_variant_name_threshold,
            struct_threshold: conf.struct_field_name_threshold,
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            allow_exact_repetitions: conf.allow_exact_repetitions,
            allow_private_module_inception: conf.allow_private_module_inception,
            allowed_prefixes: conf.allowed_prefixes.iter().map(|s| to_camel_case(s)).collect(),
        }
    }

    fn is_allowed_prefix(&self, prefix: &str) -> bool {
        self.allowed_prefixes.contains(prefix)
    }
}

impl_lint_pass!(ItemNameRepetitions => [
    ENUM_VARIANT_NAMES,
    STRUCT_FIELD_NAMES,
    MODULE_NAME_REPETITIONS,
    MODULE_INCEPTION
]);

#[must_use]
fn have_no_extra_prefix(prefixes: &[&str]) -> bool {
    prefixes.iter().all(|p| p == &"" || p == &"_")
}

impl ItemNameRepetitions {
    /// Lint the names of enum variants against the name of the enum.
    fn check_variants(&self, cx: &LateContext<'_>, item: &Item<'_>, def: &EnumDef<'_>) {
        if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(item.owner_id.def_id) {
            return;
        }

        if (def.variants.len() as u64) < self.enum_threshold {
            return;
        }

        let Some(ident) = item.kind.ident() else {
            return;
        };
        let item_name = ident.name.as_str();
        for var in def.variants {
            check_enum_start(cx, item_name, var);
            check_enum_end(cx, item_name, var);
        }

        Self::check_enum_common_affix(cx, item, def);
    }

    /// Lint the names of struct fields against the name of the struct.
    fn check_fields(&self, cx: &LateContext<'_>, item: &Item<'_>, fields: &[FieldDef<'_>]) {
        if (fields.len() as u64) < self.struct_threshold {
            return;
        }

        self.check_struct_name_repetition(cx, item, fields);
        self.check_struct_common_affix(cx, item, fields);
    }

    fn check_enum_common_affix(cx: &LateContext<'_>, item: &Item<'_>, def: &EnumDef<'_>) {
        let first = match def.variants.first() {
            Some(variant) => variant.ident.name.as_str(),
            None => return,
        };
        let mut pre = camel_case_split(first);
        let mut post = pre.clone();
        post.reverse();
        for var in def.variants {
            let name = var.ident.name.as_str();

            let variant_split = camel_case_split(name);
            if variant_split.len() == 1 {
                return;
            }

            pre = pre
                .iter()
                .zip(variant_split.iter())
                .take_while(|(a, b)| a == b)
                .map(|e| *e.0)
                .collect();
            post = post
                .iter()
                .zip(variant_split.iter().rev())
                .take_while(|(a, b)| a == b)
                .map(|e| *e.0)
                .collect();
        }
        let (what, value) = match (have_no_extra_prefix(&pre), post.is_empty()) {
            (true, true) => return,
            (false, _) => ("pre", pre.join("")),
            (true, false) => {
                post.reverse();
                ("post", post.join(""))
            },
        };
        span_lint_and_help(
            cx,
            ENUM_VARIANT_NAMES,
            item.span,
            format!("all variants have the same {what}fix: `{value}`"),
            None,
            format!(
                "remove the {what}fixes and use full paths to \
                 the variants instead of glob imports"
            ),
        );
    }

    fn check_struct_common_affix(&self, cx: &LateContext<'_>, item: &Item<'_>, fields: &[FieldDef<'_>]) {
        // if the SyntaxContext of the identifiers of the fields and struct differ dont lint them.
        // this prevents linting in macros in which the location of the field identifier names differ
        if !fields
            .iter()
            .all(|field| item.kind.ident().is_some_and(|i| i.span.eq_ctxt(field.ident.span)))
        {
            return;
        }

        if self.avoid_breaking_exported_api
            && fields
                .iter()
                .any(|field| cx.effective_visibilities.is_exported(field.def_id))
        {
            return;
        }

        let mut pre: Vec<&str> = match fields.first() {
            Some(first_field) => first_field.ident.name.as_str().split('_').collect(),
            None => return,
        };
        let mut post = pre.clone();
        post.reverse();
        for field in fields {
            let field_split: Vec<&str> = field.ident.name.as_str().split('_').collect();
            if field_split.len() == 1 {
                return;
            }

            pre = pre
                .into_iter()
                .zip(field_split.iter())
                .take_while(|(a, b)| &a == b)
                .map(|e| e.0)
                .collect();
            post = post
                .into_iter()
                .zip(field_split.iter().rev())
                .take_while(|(a, b)| &a == b)
                .map(|e| e.0)
                .collect();
        }
        let prefix = pre.join("_");
        post.reverse();
        let postfix = match post.last() {
            Some(&"") => post.join("_") + "_",
            Some(_) | None => post.join("_"),
        };
        if fields.len() > 1 {
            let (what, value) = match (
                prefix.is_empty() || prefix.chars().all(|c| c == '_'),
                postfix.is_empty(),
            ) {
                (true, true) => return,
                (false, _) => ("pre", prefix),
                (true, false) => ("post", postfix),
            };
            if fields.iter().all(|field| is_bool(field.ty)) {
                // If all fields are booleans, we don't want to emit this lint.
                return;
            }
            span_lint_and_help(
                cx,
                STRUCT_FIELD_NAMES,
                item.span,
                format!("all fields have the same {what}fix: `{value}`"),
                None,
                format!("remove the {what}fixes"),
            );
        }
    }

    fn check_struct_name_repetition(&self, cx: &LateContext<'_>, item: &Item<'_>, fields: &[FieldDef<'_>]) {
        let Some(ident) = item.kind.ident() else { return };
        let snake_name = to_snake_case(ident.name.as_str());
        let item_name_words: Vec<&str> = snake_name.split('_').collect();
        for field in fields {
            if self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(field.def_id) {
                continue;
            }

            if !field.ident.span.eq_ctxt(ident.span) {
                // consider linting only if the field identifier has the same SyntaxContext as the item(struct)
                continue;
            }

            let field_words: Vec<&str> = field.ident.name.as_str().split('_').collect();
            if field_words.len() >= item_name_words.len() {
                // if the field name is shorter than the struct name it cannot contain it
                if field_words.iter().zip(item_name_words.iter()).all(|(a, b)| a == b) {
                    span_lint_hir(
                        cx,
                        STRUCT_FIELD_NAMES,
                        field.hir_id,
                        field.span,
                        "field name starts with the struct's name",
                    );
                }
                if field_words.len() > item_name_words.len()
                    // lint only if the end is not covered by the start
                    && field_words
                        .iter()
                        .rev()
                        .zip(item_name_words.iter().rev())
                        .all(|(a, b)| a == b)
                {
                    span_lint_hir(
                        cx,
                        STRUCT_FIELD_NAMES,
                        field.hir_id,
                        field.span,
                        "field name ends with the struct's name",
                    );
                }
            }
        }
    }
}

fn check_enum_start(cx: &LateContext<'_>, item_name: &str, variant: &Variant<'_>) {
    let name = variant.ident.name.as_str();
    let item_name_chars = item_name.chars().count();

    if count_match_start(item_name, name).char_count == item_name_chars
        && name.chars().nth(item_name_chars).is_some_and(|c| !c.is_lowercase())
        && name.chars().nth(item_name_chars + 1).is_some_and(|c| !c.is_numeric())
        && !check_enum_tuple_path_match(name, variant.data)
    {
        span_lint_hir(
            cx,
            ENUM_VARIANT_NAMES,
            variant.hir_id,
            variant.span,
            "variant name starts with the enum's name",
        );
    }
}

fn check_enum_end(cx: &LateContext<'_>, item_name: &str, variant: &Variant<'_>) {
    let name = variant.ident.name.as_str();
    let item_name_chars = item_name.chars().count();

    if count_match_end(item_name, name).char_count == item_name_chars
        && !check_enum_tuple_path_match(name, variant.data)
    {
        span_lint_hir(
            cx,
            ENUM_VARIANT_NAMES,
            variant.hir_id,
            variant.span,
            "variant name ends with the enum's name",
        );
    }
}

/// Checks if an enum tuple variant contains a single field
/// whose qualified path contains the variant's name.
fn check_enum_tuple_path_match(variant_name: &str, variant_data: VariantData<'_>) -> bool {
    // Only check single-field tuple variants
    let VariantData::Tuple(fields, ..) = variant_data else {
        return false;
    };
    if fields.len() != 1 {
        return false;
    }
    // Check if field type is a path and contains the variant name
    match fields[0].ty.kind {
        TyKind::Path(QPath::Resolved(_, path)) => path
            .segments
            .iter()
            .any(|segment| segment.ident.name.as_str() == variant_name),
        TyKind::Path(QPath::TypeRelative(_, segment)) => segment.ident.name.as_str() == variant_name,
        _ => false,
    }
}

impl LateLintPass<'_> for ItemNameRepetitions {
    fn check_item_post(&mut self, _cx: &LateContext<'_>, item: &Item<'_>) {
        let Some(_ident) = item.kind.ident() else { return };

        let last = self.modules.pop();
        assert!(last.is_some());
    }

    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        let Some(ident) = item.kind.ident() else { return };

        let item_name = ident.name.as_str();
        let item_camel = to_camel_case(item_name);
        if !item.span.from_expansion() && is_present_in_source(cx, item.span)
            && let [.., (mod_name, mod_camel, mod_owner_id)] = &*self.modules
            // constants don't have surrounding modules
            && !mod_camel.is_empty()
        {
            if mod_name == &ident.name
                && let ItemKind::Mod(..) = item.kind
                && (!self.allow_private_module_inception || cx.tcx.visibility(mod_owner_id.def_id).is_public())
            {
                span_lint(
                    cx,
                    MODULE_INCEPTION,
                    item.span,
                    "module has the same name as its containing module",
                );
            }

            // The `module_name_repetitions` lint should only trigger if the item has the module in its
            // name. Having the same name is only accepted if `allow_exact_repetition` is set to `true`.

            let both_are_public =
                cx.tcx.visibility(item.owner_id).is_public() && cx.tcx.visibility(mod_owner_id.def_id).is_public();

            if both_are_public && !self.allow_exact_repetitions && item_camel == *mod_camel {
                span_lint(
                    cx,
                    MODULE_NAME_REPETITIONS,
                    ident.span,
                    "item name is the same as its containing module's name",
                );
            }

            if both_are_public && item_camel.len() > mod_camel.len() {
                let matching = count_match_start(mod_camel, &item_camel);
                let rmatching = count_match_end(mod_camel, &item_camel);
                let nchars = mod_camel.chars().count();

                let is_word_beginning = |c: char| c == '_' || c.is_uppercase() || c.is_numeric();

                if matching.char_count == nchars {
                    match item_camel.chars().nth(nchars) {
                        Some(c) if is_word_beginning(c) => span_lint(
                            cx,
                            MODULE_NAME_REPETITIONS,
                            ident.span,
                            "item name starts with its containing module's name",
                        ),
                        _ => (),
                    }
                }
                if rmatching.char_count == nchars
                    && !self.is_allowed_prefix(&item_camel[..item_camel.len() - rmatching.byte_count])
                {
                    span_lint(
                        cx,
                        MODULE_NAME_REPETITIONS,
                        ident.span,
                        "item name ends with its containing module's name",
                    );
                }
            }
        }

        if span_is_local(item.span) {
            match item.kind {
                ItemKind::Enum(_, def, _) => {
                    self.check_variants(cx, item, &def);
                },
                ItemKind::Struct(_, VariantData::Struct { fields, .. }, _) => {
                    self.check_fields(cx, item, fields);
                },
                _ => (),
            }
        }
        self.modules.push((ident.name, item_camel, item.owner_id));
    }
}
