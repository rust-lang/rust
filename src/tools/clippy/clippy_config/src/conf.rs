use crate::ConfMetadata;
use crate::de::{DeserializeOrDefault, DiagCtxt, FromDefault, create_value_list_msg, find_closest_match};
use crate::types::{
    DisallowedPath, DisallowedPathWithoutReplacement, InherentImplLintScope, MacroMatcher, MatchLintBehaviour,
    PubUnderscoreFieldsBehaviour, Rename, SourceItemOrdering, SourceItemOrderingModuleItemGroupings,
    SourceItemOrderingTraitAssocItemKinds, SourceItemOrderingWithinModuleItemGroupings, TraitImplItemOrder,
};
use rustc_attr_parsing::parse_version;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::attrs::RustcVersion;
use rustc_session::Session;
use rustc_span::{Pos as _, SourceFile, Symbol};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::{env, fs, io};
use toml::de::DeTable;

#[rustfmt::skip]
static DEFAULT_DOC_VALID_IDENTS: &[&str] = &[
    "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
    "MHz", "GHz", "THz",
    "AccessKit",
    "CoAP", "CoreFoundation", "CoreGraphics", "CoreText",
    "DevOps",
    "Direct2D", "Direct3D", "DirectWrite", "DirectX",
    "ECMAScript",
    "GPLv2", "GPLv3",
    "GitHub", "GitLab",
    "IPv4", "IPv6",
    "InfiniBand", "RoCE",
    "ClojureScript", "CoffeeScript", "JavaScript", "PostScript", "PureScript", "TypeScript",
    "PowerPC", "PowerShell", "WebAssembly",
    "NaN", "NaNs",
    "OAuth", "GraphQL",
    "SQLite", "MySQL", "PostgreSQL", "MariaDB", "MongoDB",
    "OCaml",
    "OpenAL", "OpenDNS", "OpenGL", "OpenMP", "OpenSSH", "OpenSSL", "OpenStreetMap", "OpenTelemetry",
    "OpenType",
    "WebGL", "WebGL2", "WebGPU", "WebRTC", "WebSocket", "WebTransport",
    "WebP", "OpenExr", "YCbCr", "sRGB",
    "TensorFlow",
    "TrueType",
    "iOS", "macOS", "FreeBSD", "NetBSD", "OpenBSD", "NixOS",
    "TeX", "LaTeX", "BibTeX", "BibLaTeX",
    "MinGW",
    "CamelCase",
];
static DEFAULT_DISALLOWED_NAMES: &[&str] = &["foo", "baz", "quux"];
static DEFAULT_ALLOWED_IDENTS_BELOW_MIN_CHARS: &[&str] = &["i", "j", "x", "y", "z", "w", "n"];
static DEFAULT_ALLOWED_PREFIXES: &[&str] = &["to", "as", "into", "from", "try_into", "try_from"];
static DEFAULT_ALLOWED_TRAITS_WITH_RENAMED_PARAMS: &[&str] =
    &["core::convert::From", "core::convert::TryFrom", "core::str::FromStr"];

static DEFAULT_ALLOWED_SCRIPTS: &[&str] = &["Latin"];
static DEFAULT_IGNORE_INTERIOR_MUTABILITY: &[&str] = &["bytes::Bytes"];

macro_rules! first_expr {
    ($e:expr $(,$_e:expr)*) => {
        $e
    };
}

macro_rules! filtered_names {
    (($($names:literal)*)) => { &[$($names),*] };
    (($($names:literal)*) $name:literal $($rest:tt)*) => {
        filtered_names!(($($names)* $name) $($rest)*)
    };
    (($($names:literal)*) $new_name:ident $name:literal $($rest:tt)*) => {
        filtered_names!(($($names)*) $($rest)*)
    };
}

macro_rules! define_Conf {
    (
        $(
            $(#[doc = $doc:literal])*
            $(#[default_text = $default_text:literal])?
            $(#[rename = $new_name:ident])?
            $(#[lints($($for_lints:ident),* $(,)?)])?
            // The type must exist for regular fields and shouldn't exist for deprecated ones.
            $name:ident($name_str:literal) $(: $ty:ty $(= $default:expr)?)?,
        )*
    ) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy)]
        enum ConfField {
            $($name,)*
            ThirdParty,
        }
        impl ConfField {
            const NAMES: &'static [&'static str] = &[$($name_str,)* "third-party"];
            const FIELDS: &'static [Self] = &[$(first_expr!($(Self::$new_name,)? Self::$name),)* Self::ThirdParty];
            const SUGG_NAMES: &'static [&'static str] = filtered_names!(() $($($new_name)? $name_str)* "third-party");

            fn name(self) -> &'static str {
                Self::NAMES[self as usize]
            }

            fn new_field(self) -> Self {
                Self::FIELDS[self as usize]
            }

            fn parse(s: &str) -> Option<Self> {
                match s {
                    $($name_str => Some(Self::$name),)*
                    "third-party" => Some(Self::ThirdParty),
                    _ => None,
                }
            }
        }

        /// Clippy lint configuration
        pub struct Conf {
            // TODO: emit documentation
            $($(pub $name: $ty,)?)*
        }

        impl Default for Conf {
            fn default() -> Self {
                Self {
                    $($($name: <$ty as FromDefault<_>>::from_default(first_expr!($($default,)? ())),)?)*
                }
            }
        }

        impl Conf {
            pub fn get_metadata() -> Vec<ConfMetadata> {
                vec![$(
                    ConfMetadata {
                        name: $name_str,
                        default: first_expr!(
                            $($default_text.into(),)?
                            $(<$ty as FromDefault<_>>::display_default(first_expr!($($default,)? ())).to_string(),)?
                            String::new()
                        ),
                        lints: &[$($(stringify!($for_lints)),*)?],
                        doc: concat!($($doc, '\n',)*),
                        renamed_to: first_expr!($(Some(ConfField::$new_name.name()),)? None),
                    },
                )*]
            }

            fn deserialize(dcx: &DiagCtxt<'_>, table: &toml::de::DeTable<'_>) -> Self {
                $($(let mut $name: Option<$ty> = None;)?)*

                for (key, value) in table.iter() {
                    let Some(mut conf_key) = ConfField::parse(key.get_ref()) else {
                        let sp = dcx.make_sp(key.span());
                        let mut diag = dcx.inner.struct_span_err(sp, "unknown field name");
                        if let Some(sugg) = find_closest_match(key.get_ref(), ConfField::SUGG_NAMES) {
                            diag.span_suggestion(sp, "did you mean", sugg, Applicability::MaybeIncorrect);
                        }
                        diag.note_once(create_value_list_msg(dcx, ConfField::SUGG_NAMES));
                        diag.emit();
                        continue;
                    };
                    loop {
                        match conf_key {
                            $($(ConfField::$name => {
                                // Duplicate keys are handled by the toml parser.
                                $name = Some(
                                    <$ty as DeserializeOrDefault<_>>::deserialize_or_default(
                                        dcx,
                                        value.into(),
                                        first_expr!($($default,)? ()),
                                    ),
                                );
                            },)?)*
                            ConfField::ThirdParty => {},
                            // All deprecated fields.
                            _ => {
                                let sp = dcx.make_sp(table.get_key_value(key).unwrap().0.span());
                                conf_key = conf_key.new_field();
                                let other_value = table.get_key_value(conf_key.name());
                                dcx.inner.struct_span_warn(sp, format!("use of a deprecated field"))
                                    .with_span_suggestion(
                                    sp, "use new name", conf_key.name(),
                                    if other_value.is_some() {
                                        Applicability::MaybeIncorrect
                                    } else {
                                        Applicability::MachineApplicable
                                    }
                                ).emit();

                                if let Some((other_key, _)) = other_value {
                                    dcx.inner.struct_span_err(sp, format!("duplicate key in document root"))
                                        .with_span_note(dcx.make_sp(other_key.span()), "previous definition here")
                                        .emit();
                                } else {
                                    continue;
                                }
                            },
                        }
                        break;
                    }
                }

                Self {$($(
                    $name: $name.unwrap_or_else(
                        || <$ty as FromDefault<_>>::from_default(first_expr!($($default,)? ()))
                    ),
                )?)*}
            }
        }

        #[test]
        fn check_conf_order() {
            for [x, y] in ConfField::NAMES[..ConfField::NAMES.len() - 1].array_windows::<2>() {
                assert!(x <= y, "configuration `{x}` and `{y}` are out of order");
            }
        }

        #[test]
        fn check_conf_names() {$(
            assert_eq!(stringify!($name).replace('_', "-"), $name_str);
        )*}
    };
}

define_Conf! {
    /// Which crates to allow absolute paths from
    #[lints(absolute_paths)]
    absolute_paths_allowed_crates("absolute-paths-allowed-crates"): FxHashSet<Symbol>,
    /// The maximum number of segments a path can have before being linted, anything above this will
    /// be linted.
    #[lints(absolute_paths)]
    absolute_paths_max_segments("absolute-paths-max-segments"): u64 = 2,
    /// Whether to accept a safety comment to be placed above the attributes for the `unsafe` block
    #[lints(undocumented_unsafe_blocks)]
    accept_comment_above_attributes("accept-comment-above-attributes"): bool = true,
    /// Whether to accept a safety comment to be placed above the statement containing the `unsafe` block
    #[lints(undocumented_unsafe_blocks)]
    accept_comment_above_statement("accept-comment-above-statement"): bool = true,
    /// Don't lint when comparing the result of a modulo operation to zero.
    #[lints(modulo_arithmetic)]
    allow_comparison_to_zero("allow-comparison-to-zero"): bool = true,
    /// Whether `dbg!` should be allowed in test functions or `#[cfg(test)]`
    #[lints(dbg_macro)]
    allow_dbg_in_tests("allow-dbg-in-tests"): bool = false,
    /// Whether an item should be allowed to have the same name as its containing module
    #[lints(module_name_repetitions)]
    allow_exact_repetitions("allow-exact-repetitions"): bool = true,
    /// Whether `expect` should be allowed in code always evaluated at compile time
    #[lints(expect_used)]
    allow_expect_in_consts("allow-expect-in-consts"): bool = true,
    /// Whether `expect` should be allowed in test functions or `#[cfg(test)]`
    #[lints(expect_used)]
    allow_expect_in_tests("allow-expect-in-tests"): bool = false,
    /// Whether `indexing_slicing` should be allowed in test functions or `#[cfg(test)]`
    #[lints(indexing_slicing)]
    allow_indexing_slicing_in_tests("allow-indexing-slicing-in-tests"): bool = false,
    /// Whether functions inside `#[cfg(test)]` modules or test functions should be checked.
    #[lints(large_stack_frames)]
    allow_large_stack_frames_in_tests("allow-large-stack-frames-in-tests"): bool = true,
    /// Whether to allow mixed uninlined format args, e.g. `format!("{} {}", a, foo.bar)`
    #[lints(uninlined_format_args)]
    allow_mixed_uninlined_format_args("allow-mixed-uninlined-format-args"): bool = true,
    /// Whether to allow `r#""#` when `r""` can be used
    #[lints(needless_raw_string_hashes)]
    allow_one_hash_in_raw_strings("allow-one-hash-in-raw-strings"): bool = false,
    /// Whether `panic` should be allowed in test functions or `#[cfg(test)]`
    #[lints(panic)]
    allow_panic_in_tests("allow-panic-in-tests"): bool = false,
    /// Whether print macros (ex. `println!`) should be allowed in test functions or `#[cfg(test)]`
    #[lints(print_stderr, print_stdout)]
    allow_print_in_tests("allow-print-in-tests"): bool = false,
    /// Whether to allow module inception if it's not public.
    #[lints(module_inception)]
    allow_private_module_inception("allow-private-module-inception"): bool = false,
    /// List of trait paths to ignore when checking renamed function parameters.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allow-renamed-params-for = [ "std::convert::From" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// - By default, the following traits are ignored: `From`, `TryFrom`, `FromStr`
    /// - `".."` can be used as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(renamed_function_params)]
    allow_renamed_params_for("allow-renamed-params-for"): Vec<String> = DEFAULT_ALLOWED_TRAITS_WITH_RENAMED_PARAMS,
    /// Whether `unwrap` should be allowed in code always evaluated at compile time
    #[lints(unwrap_used)]
    allow_unwrap_in_consts("allow-unwrap-in-consts"): bool = true,
    /// Whether `unwrap` should be allowed in test functions or `#[cfg(test)]`
    #[lints(unwrap_used)]
    allow_unwrap_in_tests("allow-unwrap-in-tests"): bool = false,
    /// List of types to allow `unwrap()` and `expect()` on.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allow-unwrap-types = [ "std::sync::LockResult" ]
    /// ```
    #[lints(expect_used, unwrap_used)]
    allow_unwrap_types("allow-unwrap-types"): Vec<String>,
    /// Whether `useless_vec` should ignore test functions or `#[cfg(test)]`
    #[lints(useless_vec)]
    allow_useless_vec_in_tests("allow-useless-vec-in-tests"): bool = false,
    /// Additional dotfiles (files or directories starting with a dot) to allow
    #[lints(path_ends_with_ext)]
    allowed_dotfiles("allowed-dotfiles"): Vec<String>,
    /// A list of crate names to allow duplicates of
    #[lints(multiple_crate_versions)]
    allowed_duplicate_crates("allowed-duplicate-crates"): FxHashSet<String>,
    /// Allowed names below the minimum allowed characters. The value `".."` can be used as part of
    /// the list to indicate that the configured values should be appended to the default
    /// configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(min_ident_chars)]
    allowed_idents_below_min_chars("allowed-idents-below-min-chars"): FxHashSet<String> = DEFAULT_ALLOWED_IDENTS_BELOW_MIN_CHARS,
    /// List of prefixes to allow when determining whether an item's name ends with the module's name.
    /// If the rest of an item's name is an allowed prefix (e.g. item `ToFoo` or `to_foo` in module `foo`),
    /// then don't emit a warning.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allowed-prefixes = [ "to", "from" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// - By default, the following prefixes are allowed: `to`, `as`, `into`, `from`, `try_into` and `try_from`
    /// - PascalCase variant is included automatically for each snake_case variant (e.g. if `try_into` is included,
    ///   `TryInto` will also be included)
    /// - Use `".."` as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value
    #[lints(module_name_repetitions)]
    allowed_prefixes("allowed-prefixes"): Vec<String> = DEFAULT_ALLOWED_PREFIXES,
    /// The list of unicode scripts allowed to be used in the scope.
    #[lints(disallowed_script_idents)]
    allowed_scripts("allowed-scripts"): Vec<String> = DEFAULT_ALLOWED_SCRIPTS,
    /// List of path segments allowed to have wildcard imports.
    ///
    /// #### Example
    ///
    /// ```toml
    /// allowed-wildcard-imports = [ "utils", "common" ]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// 1. This configuration has no effects if used with `warn_on_all_wildcard_imports = true`.
    /// 2. Paths with any segment that containing the word 'prelude'
    /// are already allowed by default.
    #[lints(wildcard_imports)]
    allowed_wildcard_imports("allowed-wildcard-imports"): FxHashSet<String>,
    /// Suppress checking of the passed type names in all types of operations.
    ///
    /// If a specific operation is desired, consider using `arithmetic_side_effects_allowed_binary` or `arithmetic_side_effects_allowed_unary` instead.
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed = ["SomeType", "AnotherType"]
    /// ```
    ///
    /// #### Noteworthy
    ///
    /// A type, say `SomeType`, listed in this configuration has the same behavior of
    /// `["SomeType" , "*"], ["*", "SomeType"]` in `arithmetic_side_effects_allowed_binary`.
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed("arithmetic-side-effects-allowed"): Vec<String>,
    /// Suppress checking of the passed type pair names in binary operations like addition or
    /// multiplication.
    ///
    /// Supports the "*" wildcard to indicate that a certain type won't trigger the lint regardless
    /// of the involved counterpart. For example, `["SomeType", "*"]` or `["*", "AnotherType"]`.
    ///
    /// Pairs are asymmetric, which means that `["SomeType", "AnotherType"]` is not the same as
    /// `["AnotherType", "SomeType"]`.
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed-binary = [["SomeType" , "f32"], ["AnotherType", "*"]]
    /// ```
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed_binary("arithmetic-side-effects-allowed-binary"): Vec<[String; 2]>,
    /// Suppress checking of the passed type names in unary operations like "negation" (`-`).
    ///
    /// #### Example
    ///
    /// ```toml
    /// arithmetic-side-effects-allowed-unary = ["SomeType", "AnotherType"]
    /// ```
    #[lints(arithmetic_side_effects)]
    arithmetic_side_effects_allowed_unary("arithmetic-side-effects-allowed-unary"): Vec<String>,
    /// The maximum allowed size for arrays on the stack
    #[lints(large_const_arrays, large_stack_arrays)]
    array_size_threshold("array-size-threshold"): u64 = 16 * 1024,
    /// Suppress lints whenever the suggested change would cause breakage for other crates.
    #[lints(
        box_collection,
        enum_variant_names,
        large_types_passed_by_value,
        linkedlist,
        needless_pass_by_ref_mut,
        option_option,
        owned_cow,
        rc_buffer,
        rc_mutex,
        redundant_allocation,
        ref_option,
        single_call_fn,
        trivially_copy_pass_by_ref,
        unnecessary_box_returns,
        unnecessary_wraps,
        unused_self,
        upper_case_acronyms,
        vec_box,
        wrong_self_convention,
    )]
    avoid_breaking_exported_api("avoid-breaking-exported-api"): bool = true,
    /// The list of types which may not be held across an await point.
    #[lints(await_holding_invalid_type)]
    await_holding_invalid_types("await-holding-invalid-types"): Vec<DisallowedPathWithoutReplacement>,
    #[rename = disallowed_names]
    blacklisted_names("blacklisted-names"),
    /// For internal testing only, ignores the current `publish` settings in the Cargo manifest.
    #[lints(cargo_common_metadata)]
    cargo_ignore_publish("cargo-ignore-publish"): bool = false,
    /// Whether to check for grouped late initializations from multiple `let` statements.
    ///
    /// #### Example
    /// ```rust
    /// let a;
    /// let b;
    /// if true {
    ///     a = 1;
    ///     b = 2;
    /// } else {
    ///     a = 3;
    ///     b = 4;
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// let (a, b) = if true {
    ///     (1, 2)
    /// } else {
    ///     (3, 4)
    /// };
    /// ```
    #[lints(needless_late_init)]
    check_grouped_late_init("check-grouped-late-init"): bool = true,
    /// Whether to check MSRV compatibility in `#[test]` and `#[cfg(test)]` code.
    #[lints(incompatible_msrv)]
    check_incompatible_msrv_in_tests("check-incompatible-msrv-in-tests"): bool = false,
    /// Whether to suggest reordering constructor fields when initializers are present.
    ///
    /// Warnings produced by this configuration aren't necessarily fixed by just reordering the fields. Even if the
    /// suggested code would compile, it can change semantics if the initializer expressions have side effects. The
    /// following example [from rust-clippy#11846] shows how the suggestion can run into borrow check errors:
    ///
    /// ```rust
    /// struct MyStruct {
    ///     vector: Vec<u32>,
    ///     length: usize
    /// }
    /// fn main() {
    ///     let vector = vec![1,2,3];
    ///     MyStruct { length: vector.len(), vector};
    /// }
    /// ```
    ///
    /// [from rust-clippy#11846]: https://github.com/rust-lang/rust-clippy/issues/11846#issuecomment-1820747924
    #[lints(inconsistent_struct_constructor)]
    check_inconsistent_struct_field_initializers("check-inconsistent-struct-field-initializers"): bool = false,
    /// Whether to also run the listed lints on private items.
    #[lints(missing_errors_doc, missing_panics_doc, missing_safety_doc, unnecessary_safety_doc)]
    check_private_items("check-private-items"): bool = false,
    /// The maximum cognitive complexity a function can have
    #[lints(cognitive_complexity)]
    cognitive_complexity_threshold("cognitive-complexity-threshold"): u64 = 25,
    /// The minimum digits a const float literal must have to supress the `excessive_precicion` lint
    #[lints(excessive_precision)]
    const_literal_digits_threshold("const-literal-digits-threshold"): u32 = 30,
    #[rename = cognitive_complexity_threshold]
    cyclomatic_complexity_threshold("cyclomatic-complexity-threshold"),
    /// The list of disallowed fields, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the field that should be disallowed
    /// - `reason` (optional): explanation why this field is disallowed
    /// - `replacement` (optional): suggested alternative method
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[lints(disallowed_fields)]
    disallowed_fields("disallowed-fields"): Vec<DisallowedPath>,
    /// The list of disallowed macros, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the macro that should be disallowed
    /// - `reason` (optional): explanation why this macro is disallowed
    /// - `replacement` (optional): suggested alternative macro
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[lints(disallowed_macros)]
    disallowed_macros("disallowed-macros"): Vec<DisallowedPath>,
    /// The list of disallowed methods, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the method that should be disallowed
    /// - `reason` (optional): explanation why this method is disallowed
    /// - `replacement` (optional): suggested alternative method
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[lints(disallowed_methods)]
    disallowed_methods("disallowed-methods"): Vec<DisallowedPath>,
    /// The list of disallowed names to lint about. NB: `bar` is not here since it has legitimate uses. The value
    /// `".."` can be used as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value.
    #[lints(disallowed_names)]
    disallowed_names("disallowed-names"): Vec<String> = DEFAULT_DISALLOWED_NAMES,
    /// The list of disallowed types, written as fully qualified paths.
    ///
    /// **Fields:**
    /// - `path` (required): the fully qualified path to the type that should be disallowed
    /// - `reason` (optional): explanation why this type is disallowed
    /// - `replacement` (optional): suggested alternative type
    /// - `allow-invalid` (optional, `false` by default): when set to `true`, it will ignore this entry
    ///   if the path doesn't exist, instead of emitting an error
    #[lints(disallowed_types)]
    disallowed_types("disallowed-types"): Vec<DisallowedPath>,
    /// The list of words this lint should not consider as identifiers needing ticks. The value
    /// `".."` can be used as part of the list to indicate that the configured values should be appended to the
    /// default configuration of Clippy. By default, any configuration will replace the default value. For example:
    /// * `doc-valid-idents = ["ClipPy"]` would replace the default list with `["ClipPy"]`.
    /// * `doc-valid-idents = ["ClipPy", ".."]` would append `ClipPy` to the default list.
    #[lints(doc_markdown)]
    doc_valid_idents("doc-valid-idents"): FxHashSet<String> = DEFAULT_DOC_VALID_IDENTS,
    /// Whether to apply the raw pointer heuristic to determine if a type is `Send`.
    #[lints(non_send_fields_in_send_ty)]
    enable_raw_pointer_heuristic_for_send("enable-raw-pointer-heuristic-for-send"): bool = true,
    /// Whether to recommend using implicit into iter for reborrowed values.
    ///
    /// #### Example
    /// ```no_run
    /// let mut vec = vec![1, 2, 3];
    /// let rmvec = &mut vec;
    /// for _ in rmvec.iter() {}
    /// for _ in rmvec.iter_mut() {}
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// let mut vec = vec![1, 2, 3];
    /// let rmvec = &mut vec;
    /// for _ in &*rmvec {}
    /// for _ in &mut *rmvec {}
    /// ```
    #[lints(explicit_iter_loop)]
    enforce_iter_loop_reborrow("enforce-iter-loop-reborrow"): bool = false,
    /// The list of imports to always rename, a fully qualified path followed by the rename.
    #[lints(missing_enforced_import_renames)]
    enforced_import_renames("enforced-import-renames"): Vec<Rename>,
    /// The minimum number of enum variants for the lints about variant names to trigger
    #[lints(enum_variant_names)]
    enum_variant_name_threshold("enum-variant-name-threshold"): u64 = 3,
    /// The maximum size of an enum's variant to avoid box suggestion
    #[lints(large_enum_variant)]
    enum_variant_size_threshold("enum-variant-size-threshold"): u64 = 200,
    /// The maximum amount of nesting a block can reside in
    #[lints(excessive_nesting)]
    excessive_nesting_threshold("excessive-nesting-threshold"): u64 = 0,
    /// The maximum byte size a `Future` can have, before it triggers the `clippy::large_futures` lint
    #[lints(large_futures)]
    future_size_threshold("future-size-threshold"): u64 = 16 * 1024,
    /// A list of paths to types that should be treated as if they do not contain interior mutability
    #[lints(borrow_interior_mutable_const, declare_interior_mutable_const, ifs_same_cond, mutable_key_type)]
    ignore_interior_mutability("ignore-interior-mutability"): Vec<String> = DEFAULT_IGNORE_INTERIOR_MUTABILITY,
    /// Sets the scope ("crate", "file", or "module") in which duplicate inherent `impl` blocks for the same type are linted.
    #[lints(multiple_inherent_impl)]
    inherent_impl_lint_scope("inherent-impl-lint-scope"): InherentImplLintScope = InherentImplLintScope::Crate,
    /// A list of paths to types that should be ignored as overly large `Err`-variants in a
    /// `Result` returned from a function
    #[lints(result_large_err)]
    large_error_ignored("large-error-ignored"): Vec<String>,
    /// The maximum size of the `Err`-variant in a `Result` returned from a function
    #[lints(result_large_err)]
    large_error_threshold("large-error-threshold"): u64 = 128,
    /// Whether collapsible `if` and `else if` chains are linted if they contain comments inside the parts
    /// that would be collapsed.
    #[lints(collapsible_else_if, collapsible_if)]
    lint_commented_code("lint-commented-code"): bool = false,
    #[rename = check_inconsistent_struct_field_initializers]
    lint_inconsistent_struct_field_initializers("lint-inconsistent-struct-field-initializers"): bool = false,
    /// The lower bound for linting decimal literals
    #[lints(decimal_literal_representation)]
    literal_representation_threshold("literal-representation-threshold"): u64 = 16384,
    /// Whether the matches should be considered by the lint, and whether there should
    /// be filtering for common types.
    #[lints(manual_let_else)]
    matches_for_let_else("matches-for-let-else"): MatchLintBehaviour = MatchLintBehaviour::WellKnownTypes,
    /// The maximum number of bool parameters a function can have.
    /// Use `0` to lint on any function with a bool parameter.
    #[lints(fn_params_excessive_bools)]
    max_fn_params_bools("max-fn-params-bools"): u64 = 3,
    /// The maximum size of a file included via `include_bytes!()` or `include_str!()`, in bytes
    #[lints(large_include_file)]
    max_include_file_size("max-include-file-size"): u64 = 1_000_000,
    /// The maximum number of bool fields a struct can have
    #[lints(struct_excessive_bools)]
    max_struct_bools("max-struct-bools"): u64 = 3,
    /// When Clippy suggests using a slice pattern, this is the maximum number of elements allowed in
    /// the slice pattern that is suggested. If more elements are necessary, the lint is suppressed.
    /// For example, `[_, _, _, e, ..]` is a slice pattern with 4 elements.
    #[lints(index_refutable_slice)]
    max_suggested_slice_pattern_length("max-suggested-slice-pattern-length"): u64 = 3,
    /// The maximum number of bounds a trait can have to be linted
    #[lints(type_repetition_in_bounds)]
    max_trait_bounds("max-trait-bounds"): u64 = 3,
    /// Whether to lint idents that have too few chars even when following trait declaration.
    #[lints(min_ident_chars)]
    min_ident_chars_lint_trait_impl("min-ident-chars-lint-trait-impl"): bool = false,
    /// Minimum chars an ident can have, anything below or equal to this will be linted.
    #[lints(min_ident_chars)]
    min_ident_chars_threshold("min-ident-chars-threshold"): u64 = 1,
    /// Whether to allow fields starting with an underscore to skip documentation requirements
    #[lints(missing_docs_in_private_items)]
    missing_docs_allow_unused("missing-docs-allow-unused"): bool = false,
    /// Whether to **only** check for missing documentation in items visible within the current
    /// crate. For example, `pub(crate)` items.
    #[lints(missing_docs_in_private_items)]
    missing_docs_in_crate_items("missing-docs-in-crate-items"): bool = false,
    /// The named groupings of different source item kinds within modules.
    #[lints(arbitrary_source_item_ordering)]
    module_item_order_groupings("module-item-order-groupings"): SourceItemOrderingModuleItemGroupings,
    /// Whether the items within module groups should be ordered alphabetically or not.
    ///
    /// This option can be configured to "all", "none", or a list of specific grouping names that should be checked
    /// (e.g. only "enums").
    #[lints(arbitrary_source_item_ordering)]
    module_items_ordered_within_groupings("module-items-ordered-within-groupings"): SourceItemOrderingWithinModuleItemGroupings,
    /// The minimum rust version that the project supports. Defaults to the `rust-version` field in `Cargo.toml`
    #[default_text = "current version"]
    #[lints(
        allow_attributes,
        allow_attributes_without_reason,
        almost_complete_range,
        approx_constant,
        assigning_clones,
        borrow_as_ptr,
        cast_abs_to_unsigned,
        checked_conversions,
        cloned_instead_of_copied,
        collapsible_match,
        collapsible_str_replace,
        deprecated_cfg_attr,
        derivable_impls,
        err_expect,
        filter_map_next,
        from_over_into,
        if_then_some_else_none,
        implicit_saturating_sub,
        index_refutable_slice,
        inefficient_to_string,
        io_other_error,
        iter_kv_map,
        legacy_numeric_constants,
        len_zero,
        lines_filter_map_ok,
        manual_abs_diff,
        manual_bits,
        manual_c_str_literals,
        manual_clamp,
        manual_div_ceil,
        manual_flatten,
        manual_hash_one,
        manual_is_ascii_check,
        manual_is_power_of_two,
        manual_is_variant_and,
        manual_isolate_lowest_one,
        manual_let_else,
        manual_midpoint,
        manual_non_exhaustive,
        manual_noop_waker,
        manual_option_as_slice,
        manual_pattern_char_comparison,
        manual_range_contains,
        manual_rem_euclid,
        manual_repeat_n,
        manual_retain,
        manual_slice_fill,
        manual_slice_size_calculation,
        manual_split_once,
        manual_str_repeat,
        manual_strip,
        manual_take,
        manual_try_fold,
        map_clone,
        map_unwrap_or,
        map_with_unused_argument_over_ranges,
        match_like_matches_macro,
        mem_replace_option_with_some,
        mem_replace_with_default,
        missing_const_for_fn,
        needless_borrow,
        non_std_lazy_statics,
        nonnull_unchecked_on_box_ptr,
        option_as_ref_deref,
        or_fun_call,
        ptr_as_ptr,
        question_mark,
        redundant_field_names,
        redundant_static_lifetimes,
        repeat_vec_with_capacity,
        same_item_push,
        seek_from_current,
        to_digit_is_some,
        transmute_ptr_to_ref,
        tuple_array_conversions,
        type_repetition_in_bounds,
        unchecked_time_subtraction,
        uninlined_format_args,
        unnecessary_lazy_evaluations,
        unnecessary_unwrap,
        unnested_or_patterns,
        unused_trait_names,
        use_self,
        zero_ptr,
    )]
    msrv("msrv"): Option<RustcVersion>,
    /// The minimum size (in bytes) to consider a type for passing by reference instead of by value.
    #[lints(large_types_passed_by_value)]
    pass_by_value_size_limit("pass-by-value-size-limit"): u64 = 256,
    /// Lint "public" fields in a struct that are prefixed with an underscore based on their
    /// exported visibility, or whether they are marked as "pub".
    #[lints(pub_underscore_fields)]
    pub_underscore_fields_behavior("pub-underscore-fields-behavior"): PubUnderscoreFieldsBehaviour = PubUnderscoreFieldsBehaviour::PubliclyExported,
    /// Whether the type itself in a struct or enum should be replaced with `Self` when encountering recursive types.
    #[lints(use_self)]
    recursive_self_in_type_definitions("recursive-self-in-type-definitions"): bool = true,
    /// Whether to lint only if it's multiline.
    #[lints(semicolon_inside_block)]
    semicolon_inside_block_ignore_singleline("semicolon-inside-block-ignore-singleline"): bool = false,
    /// Whether to lint only if it's singleline.
    #[lints(semicolon_outside_block)]
    semicolon_outside_block_ignore_multiline("semicolon-outside-block-ignore-multiline"): bool = false,
    /// The maximum number of single char bindings a scope may have
    #[lints(many_single_char_names)]
    single_char_binding_names_threshold("single-char-binding-names-threshold"): u64 = 4,
    /// Which kind of elements should be ordered internally, possible values being `enum`, `impl`, `module`, `struct`, `trait`.
    #[lints(arbitrary_source_item_ordering)]
    source_item_ordering("source-item-ordering"): SourceItemOrdering,
    /// The maximum allowed stack size for functions in bytes
    #[lints(large_stack_frames)]
    stack_size_threshold("stack-size-threshold"): u64 = 512_000,
    /// Enforce the named macros always use the braces specified.
    ///
    /// A `MacroMatcher` can be added like so `{ name = "macro_name", brace = "(" }`. If the macro
    /// could be used with a full path two `MacroMatcher`s have to be added one with the full path
    /// `crate_name::macro_name` and one with just the macro name.
    #[lints(nonstandard_macro_braces)]
    standard_macro_braces("standard-macro-braces"): Vec<MacroMatcher>,
    /// The minimum number of struct fields for the lints about field names to trigger
    #[lints(struct_field_names)]
    struct_field_name_threshold("struct-field-name-threshold"): u64 = 3,
    /// Whether to suppress a restriction lint in constant code. In same
    /// cases the restructured operation might not be unavoidable, as the
    /// suggested counterparts are unavailable in constant code. This
    /// configuration will cause restriction lints to trigger even
    /// if no suggestion can be made.
    #[lints(indexing_slicing)]
    suppress_restriction_lint_in_const("suppress-restriction-lint-in-const"): bool = false,
    /// The maximum size of objects (in bytes) that will be linted. Larger objects are ok on the heap
    #[lints(boxed_local, useless_vec)]
    too_large_for_stack("too-large-for-stack"): u64 = 200,
    /// The maximum number of argument a function or method can have
    #[lints(too_many_arguments)]
    too_many_arguments_threshold("too-many-arguments-threshold"): u64 = 7,
    /// The maximum number of lines a function or method can have
    #[lints(too_many_lines)]
    too_many_lines_threshold("too-many-lines-threshold"): u64 = 100,
    /// The order of associated items in traits.
    #[lints(arbitrary_source_item_ordering)]
    trait_assoc_item_kinds_order("trait-assoc-item-kinds-order"): SourceItemOrderingTraitAssocItemKinds,
    /// The required ordering of associated items in trait impls: purely alphabetical,
    /// following the trait definition order, or accepting either.
    ///
    /// Note that the trait definition order may change between versions of the
    /// crate defining the trait without being considered a breaking change.
    ///
    /// Examples:
    /// When using trait definition item ordering:
    /// ```toml
    /// trait-impl-item-order = "trait_item_ordering"
    /// ```
    /// When using trait definition item ordering and alphabetical for fallbacks:
    /// ```toml
    /// trait-impl-item-order = "alphabetical_or_trait_item_ordering"
    /// ```
    #[lints(arbitrary_source_item_ordering)]
    trait_impl_item_order("trait-impl-item-order"): TraitImplItemOrder,
    /// The maximum size (in bytes) to consider a `Copy` type for passing by value instead of by
    /// reference.
    #[default_text = "target_pointer_width"]
    #[lints(trivially_copy_pass_by_ref)]
    trivial_copy_size_limit("trivial-copy-size-limit"): Option<u64>,
    /// The maximum complexity a type can have
    #[lints(type_complexity)]
    type_complexity_threshold("type-complexity-threshold"): u64 = 250,
    /// The byte size a `T` in `Box<T>` can have, below which it triggers the `clippy::unnecessary_box` lint
    #[lints(unnecessary_box_returns)]
    unnecessary_box_size("unnecessary-box-size"): u64 = 128,
    /// Should the fraction of a decimal be linted to include separators.
    #[lints(unreadable_literal)]
    unreadable_literal_lint_fractions("unreadable-literal-lint-fractions"): bool = true,
    /// Enables verbose mode. Triggers if there is more than one uppercase char next to each other
    #[lints(upper_case_acronyms)]
    upper_case_acronyms_aggressive("upper-case-acronyms-aggressive"): bool = false,
    /// The size of the boxed type in bytes, where boxing in a `Vec` is allowed
    #[lints(vec_box)]
    vec_box_size_threshold("vec-box-size-threshold"): u64 = 4096,
    /// The maximum allowed size of a bit mask before suggesting to use 'trailing_zeros'
    #[lints(verbose_bit_mask)]
    verbose_bit_mask_threshold("verbose-bit-mask-threshold"): u64 = 1,
    /// Whether to emit warnings on all wildcard imports, including those from `prelude`, from `super` in tests,
    /// or for `pub use` reexports.
    #[lints(wildcard_imports)]
    warn_on_all_wildcard_imports("warn-on-all-wildcard-imports"): bool = false,
    /// Whether to also emit warnings for unsafe blocks with metavariable expansions in **private** macros.
    #[lints(macro_metavars_in_unsafe)]
    warn_unsafe_macro_metavars_in_private_macros("warn-unsafe-macro-metavars-in-private-macros"): bool = false,
}

// Remove code tags and code behind '# 's, as they are not needed for the lint docs and --explain
pub fn sanitize_explanation(raw_docs: &str) -> String {
    // Remove tags and hidden code:
    let mut explanation = String::with_capacity(128);
    let mut in_code = false;
    for line in raw_docs.lines() {
        let line = line.strip_prefix(' ').unwrap_or(line);

        if let Some(lang) = line.strip_prefix("```") {
            let tag = lang.split_once(',').map_or(lang, |(left, _)| left);
            if !in_code && matches!(tag, "" | "rust" | "ignore" | "should_panic" | "no_run" | "compile_fail") {
                explanation += "```rust\n";
            } else {
                explanation += line;
                explanation.push('\n');
            }
            in_code = !in_code;
        } else if !(in_code && line.starts_with("# ")) {
            explanation += line;
            explanation.push('\n');
        }
    }

    explanation
}

/// Searches for and loads the config file into the source map.
///
/// # Errors
///
/// Returns any unexpected filesystem error encountered when searching for the config file
fn load_conf_file(sess: &Session) -> Option<Arc<SourceFile>> {
    /// Possible filename to search for.
    const CONFIG_FILE_NAMES: [&str; 2] = [".clippy.toml", "clippy.toml"];

    // Start looking for a config file in CLIPPY_CONF_DIR, or failing that, CARGO_MANIFEST_DIR.
    // If neither of those exist, use ".". (Update documentation if this priority changes)
    const CONFIG_VARS: [(&str, &str); 2] = [
        ("CLIPPY_CONF_DIR", "failed to read `CLIPPY_CONF_DIR` as a directory"),
        (
            "CARGO_MANIFEST_DIR",
            "failed to read `CARGO_MANIFEST_DIR` as a directory",
        ),
    ];
    let (current, msg) = CONFIG_VARS
        .into_iter()
        .find_map(|(var, msg)| env::var_os(var).map(|p| (PathBuf::from(p), msg)))
        .unwrap_or_else(|| (PathBuf::from("."), "failed to get the current directory"));
    let mut current = match current.canonicalize() {
        Ok(x) => x,
        Err(e) => {
            sess.dcx().err(format!("{msg}: {e}"));
            return None;
        },
    };

    let mut loaded_config: Option<(PathBuf, Arc<SourceFile>)> = None;
    loop {
        for config_file_name in CONFIG_FILE_NAMES {
            if let Ok(config_path) = current.join(config_file_name).canonicalize() {
                if let Some((loaded_path, _)) = &loaded_config {
                    if fs::metadata(loaded_path).is_ok_and(|x| x.is_file()) {
                        // Warn if `.clippy.toml` and `clippy.toml` exist
                        sess.dcx().warn(format!(
                            "using config file `{}`, `{}` will be ignored",
                            loaded_path.display(),
                            config_path.display(),
                        ));
                    }
                } else {
                    match sess.source_map().load_file(&config_path) {
                        Ok(src) => loaded_config = Some((config_path, src)),
                        Err(e)
                            if matches!(
                                e.kind(),
                                io::ErrorKind::NotFound | io::ErrorKind::IsADirectory | io::ErrorKind::NotADirectory
                            ) => {},
                        Err(e) => {
                            sess.dcx()
                                .err(format!("error reading `{}`: {e}", config_path.display()));
                            return None;
                        },
                    }
                }
            }
        }

        // Don't mention config files in parent directories.
        if let Some((_, src)) = loaded_config {
            return Some(src);
        }

        // If the current directory has no parent, we're done searching.
        if !current.pop() {
            return None;
        }
    }
}

impl Conf {
    pub fn load(sess: &Session) -> &'static Conf {
        static CONF: OnceLock<Conf> = OnceLock::new();
        CONF.get_or_init(|| Conf::load_inner(sess))
    }

    fn load_inner(sess: &Session) -> Conf {
        let mut conf = if let Some(src) = load_conf_file(sess) {
            let dcx = DiagCtxt::new(sess, src.start_pos.to_usize());
            let src = src.src.as_ref().unwrap();

            let (toml, errs) = DeTable::parse_recoverable(src.as_str());
            for e in errs {
                match e.span() {
                    Some(sp) => dcx.span_err(sp, e.message().to_owned()),
                    None => {
                        dcx.inner
                            .struct_err(format!("error parsing `clippy.toml`: {}", e.message()))
                            .emit();
                    },
                }
            }
            Conf::deserialize(&dcx, toml.get_ref())
        } else {
            Conf::default()
        };

        let cargo_msrv = env::var("CARGO_PKG_RUST_VERSION")
            .ok()
            .and_then(|v| parse_version(Symbol::intern(&v)));
        match (&conf.msrv, cargo_msrv) {
            (None, Some(cargo_msrv)) => conf.msrv = Some(cargo_msrv),
            (Some(clippy_msrv), Some(cargo_msrv)) => {
                if *clippy_msrv != cargo_msrv {
                    sess.dcx().warn(format!(
                        "the MSRV in `clippy.toml` and `Cargo.toml` differ; using `{clippy_msrv}` from `clippy.toml`"
                    ));
                }
            },
            (_, None) => {},
        }

        conf
    }
}

#[cfg(test)]
mod tests {
    use rustc_data_structures::fx::FxHashSet;
    use std::fs;
    use toml::de::DeTable;
    use walkdir::WalkDir;

    #[test]
    fn configs_are_tested() {
        let mut names: FxHashSet<_> = super::Conf::get_metadata()
            .into_iter()
            .filter(|meta| meta.renamed_to.is_none())
            .map(|meta| meta.name)
            .collect();

        let toml_files = WalkDir::new("../tests")
            .into_iter()
            .map(Result::unwrap)
            .filter(|entry| entry.file_name() == "clippy.toml");

        for entry in toml_files {
            let file = fs::read_to_string(entry.path()).unwrap();
            if let Ok(toml) = DeTable::parse(&file) {
                for (key, _) in toml.as_ref() {
                    names.remove(&**key.get_ref());
                }
            }
        }

        assert!(
            names.is_empty(),
            "Configuration variable lacks test: {names:?}\nAdd a test to `tests/ui-toml`"
        );
    }
}
