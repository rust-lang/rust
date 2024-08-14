use crate::config::file_lines::FileLines;
use crate::config::macro_names::MacroSelectors;
use crate::config::options::{IgnoreList, WidthHeuristics};

/// Trait for types that can be used in `Config`.
pub(crate) trait ConfigType: Sized {
    /// Returns hint text for use in `Config::print_docs()`. For enum types, this is a
    /// pipe-separated list of variants; for other types it returns `<type>`.
    fn doc_hint() -> String;

    /// Return `true` if the variant (i.e. value of this type) is stable.
    ///
    /// By default, return true for all values. Enums annotated with `#[config_type]`
    /// are automatically implemented, based on the `#[unstable_variant]` annotation.
    fn stable_variant(&self) -> bool {
        true
    }
}

impl ConfigType for bool {
    fn doc_hint() -> String {
        String::from("<boolean>")
    }
}

impl ConfigType for usize {
    fn doc_hint() -> String {
        String::from("<unsigned integer>")
    }
}

impl ConfigType for isize {
    fn doc_hint() -> String {
        String::from("<signed integer>")
    }
}

impl ConfigType for String {
    fn doc_hint() -> String {
        String::from("<string>")
    }
}

impl ConfigType for FileLines {
    fn doc_hint() -> String {
        String::from("<json>")
    }
}

impl ConfigType for MacroSelectors {
    fn doc_hint() -> String {
        String::from("[<string>, ...]")
    }
}

impl ConfigType for WidthHeuristics {
    fn doc_hint() -> String {
        String::new()
    }
}

impl ConfigType for IgnoreList {
    fn doc_hint() -> String {
        String::from("[<string>,..]")
    }
}

macro_rules! create_config {
    // Options passed into the macro.
    //
    // - $i: the ident name of the option
    // - $ty: the type of the option value
    // - $stb: true if the option is stable
    // - $dstring: description of the option
    ($($i:ident: $ty:ty, $stb:expr, $( $dstring:expr ),+ );+ $(;)*) => (
        #[cfg(test)]
        use std::collections::HashSet;
        use std::io::Write;

        use serde::{Deserialize, Serialize};
        use $crate::config::style_edition::StyleEditionDefault;

        #[derive(Clone)]
        #[allow(unreachable_pub)]
        pub struct Config {
            // For each config item, we store:
            //
            // - 0: true if the value has been access
            // - 1: true if the option was manually initialized
            // - 2: the option value
            // - 3: true if the option is unstable
            // - 4: true if the option was set manually from a CLI flag
            // FIXME: 4 is probably unnecessary and duplicative
            // https://github.com/rust-lang/rustfmt/issues/6252
            $($i: (Cell<bool>, bool, <$ty as StyleEditionDefault>::ConfigType, bool, bool)),+
        }

        // Just like the Config struct but with each property wrapped
        // as Option<T>. This is used to parse a rustfmt.toml that doesn't
        // specify all properties of `Config`.
        // We first parse into `PartialConfig`, then create a default `Config`
        // and overwrite the properties with corresponding values from `PartialConfig`.
        #[derive(Deserialize, Serialize, Clone)]
        #[allow(unreachable_pub)]
        pub struct PartialConfig {
            $(pub $i: Option<<$ty as StyleEditionDefault>::ConfigType>),+
        }

        // Macro hygiene won't allow us to make `set_$i()` methods on Config
        // for each item, so this struct is used to give the API to set values:
        // `config.set().option(false)`. It's pretty ugly. Consider replacing
        // with `config.set_option(false)` if we ever get a stable/usable
        // `concat_idents!()`.
        #[allow(unreachable_pub)]
        pub struct ConfigSetter<'a>(&'a mut Config);

        impl<'a> ConfigSetter<'a> {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&mut self, value: <$ty as StyleEditionDefault>::ConfigType) {
                (self.0).$i.2 = value;
                match stringify!($i) {
                    "max_width"
                    | "use_small_heuristics"
                    | "fn_call_width"
                    | "single_line_if_else_max_width"
                    | "single_line_let_else_max_width"
                    | "attr_fn_like_width"
                    | "struct_lit_width"
                    | "struct_variant_width"
                    | "array_width"
                    | "chain_width" => self.0.set_heuristics(),
                    "merge_imports" => self.0.set_merge_imports(),
                    "fn_args_layout" => self.0.set_fn_args_layout(),
                    "hide_parse_errors" => self.0.set_hide_parse_errors(),
                    "version" => self.0.set_version(),
                    &_ => (),
                }
            }
            )+
        }

        #[allow(unreachable_pub)]
        pub struct CliConfigSetter<'a>(&'a mut Config);

        impl<'a> CliConfigSetter<'a> {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&mut self, value: <$ty as StyleEditionDefault>::ConfigType) {
                (self.0).$i.2 = value;
                (self.0).$i.4 = true;
                match stringify!($i) {
                    "max_width"
                    | "use_small_heuristics"
                    | "fn_call_width"
                    | "single_line_if_else_max_width"
                    | "single_line_let_else_max_width"
                    | "attr_fn_like_width"
                    | "struct_lit_width"
                    | "struct_variant_width"
                    | "array_width"
                    | "chain_width" => self.0.set_heuristics(),
                    "merge_imports" => self.0.set_merge_imports(),
                    "fn_args_layout" => self.0.set_fn_args_layout(),
                    "hide_parse_errors" => self.0.set_hide_parse_errors(),
                    "version" => self.0.set_version(),
                    &_ => (),
                }
            }
            )+
        }

        // Query each option, returns true if the user set the option, false if
        // a default was used.
        #[allow(unreachable_pub)]
        pub struct ConfigWasSet<'a>(&'a Config);

        impl<'a> ConfigWasSet<'a> {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&self) -> bool {
                (self.0).$i.1
            }
            )+
        }

        // Query each option, returns true if the user set the option via a CLI flag,
        // false if a default was used.
        #[allow(unreachable_pub)]
        pub struct CliConfigWasSet<'a>(&'a Config);

        impl<'a> CliConfigWasSet<'a> {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&self) -> bool {
                (self.0).$i.4
            }
            )+
        }

        impl Config {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&self) -> <$ty as StyleEditionDefault>::ConfigType {
                self.$i.0.set(true);
                self.$i.2.clone()
            }
            )+

            #[allow(unreachable_pub)]
            pub(super) fn default_with_style_edition(style_edition: StyleEdition) -> Config {
                Config {
                    $(
                        $i: (
                                Cell::new(false),
                                false,
                                <$ty as StyleEditionDefault>::style_edition_default(
                                    style_edition
                                ),
                                $stb,
                                false,
                            ),
                    )+
                }
            }

            #[allow(unreachable_pub)]
            pub fn set(&mut self) -> ConfigSetter<'_> {
                ConfigSetter(self)
            }

            #[allow(unreachable_pub)]
            pub fn set_cli(&mut self) -> CliConfigSetter<'_> {
                CliConfigSetter(self)
            }

            #[allow(unreachable_pub)]
            pub fn was_set(&self) -> ConfigWasSet<'_> {
                ConfigWasSet(self)
            }

            #[allow(unreachable_pub)]
            pub fn was_set_cli(&self) -> CliConfigWasSet<'_> {
                CliConfigWasSet(self)
            }

            fn fill_from_parsed_config(mut self, parsed: PartialConfig, dir: &Path) -> Config {
            $(
                if let Some(option_value) = parsed.$i {
                    let option_stable = self.$i.3;
                    if $crate::config::config_type::is_stable_option_and_value(
                        stringify!($i), option_stable, &option_value
                    ) {
                        self.$i.1 = true;
                        self.$i.2 = option_value;
                    }
                }
            )+
                self.set_heuristics();
                self.set_ignore(dir);
                self.set_merge_imports();
                self.set_fn_args_layout();
                self.set_hide_parse_errors();
                self.set_version();
                self
            }

            /// Returns a hash set initialized with every user-facing config option name.
            #[cfg(test)]
            pub(crate) fn hash_set() -> HashSet<String> {
                let mut hash_set = HashSet::new();
                $(
                    hash_set.insert(stringify!($i).to_owned());
                )+
                hash_set
            }

            pub(crate) fn is_valid_name(name: &str) -> bool {
                match name {
                    $(
                        stringify!($i) => true,
                    )+
                        _ => false,
                }
            }

            #[allow(unreachable_pub)]
            pub fn is_valid_key_val(key: &str, val: &str) -> bool {
                match key {
                    $(
                        stringify!($i) => {
                            val.parse::<<$ty as StyleEditionDefault>::ConfigType>().is_ok()
                        }
                    )+
                        _ => false,
                }
            }

            #[allow(unreachable_pub)]
            pub fn used_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: if self.$i.0.get() {
                                Some(self.$i.2.clone())
                            } else {
                                None
                            },
                    )+
                }
            }

            #[allow(unreachable_pub)]
            pub fn all_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: Some(self.$i.2.clone()),
                    )+
                }
            }

            #[allow(unreachable_pub)]
            pub fn override_value(&mut self, key: &str, val: &str)
            {
                match key {
                    $(
                        stringify!($i) => {
                            let value = val.parse::<<$ty as StyleEditionDefault>::ConfigType>()
                                .expect(
                                    &format!(
                                        "Failed to parse override for {} (\"{}\") as a {}",
                                        stringify!($i),
                                        val,
                                        stringify!(<$ty as StyleEditionDefault>::ConfigType)
                                    )
                                );

                            // Users are currently allowed to set unstable
                            // options/variants via the `--config` options override.
                            //
                            // There is ongoing discussion about how to move forward here:
                            // https://github.com/rust-lang/rustfmt/pull/5379
                            //
                            // For now, do not validate whether the option or value is stable,
                            // just always set it.
                            self.$i.1 = true;
                            self.$i.2 = value;
                        }
                    )+
                    _ => panic!("Unknown config key in override: {}", key)
                }

                match key {
                    "max_width"
                    | "use_small_heuristics"
                    | "fn_call_width"
                    | "single_line_if_else_max_width"
                    | "single_line_let_else_max_width"
                    | "attr_fn_like_width"
                    | "struct_lit_width"
                    | "struct_variant_width"
                    | "array_width"
                    | "chain_width" => self.set_heuristics(),
                    "merge_imports" => self.set_merge_imports(),
                    "fn_args_layout" => self.set_fn_args_layout(),
                    "hide_parse_errors" => self.set_hide_parse_errors(),
                    "version" => self.set_version(),
                    &_ => (),
                }
            }

            #[allow(unreachable_pub)]
            pub fn is_hidden_option(name: &str) -> bool {
                const HIDE_OPTIONS: [&str; 7] = [
                    "verbose",
                    "verbose_diff",
                    "file_lines",
                    "width_heuristics",
                    "merge_imports",
                    "fn_args_layout",
                    "hide_parse_errors"
                ];
                HIDE_OPTIONS.contains(&name)
            }

            #[allow(unreachable_pub)]
            pub fn print_docs(out: &mut dyn Write, include_unstable: bool) {
                let style_edition = StyleEdition::Edition2015;
                use std::cmp;
                let max = 0;
                $( let max = cmp::max(max, stringify!($i).len()+1); )+
                let space_str = " ".repeat(max);
                writeln!(out, "Configuration Options:").unwrap();
                $(
                    if $stb || include_unstable {
                        let name_raw = stringify!($i);

                        if !Config::is_hidden_option(name_raw) {
                            let mut name_out = String::with_capacity(max);
                            for _ in name_raw.len()..max-1 {
                                name_out.push(' ')
                            }
                            name_out.push_str(name_raw);
                            name_out.push(' ');
                            let default_value = <$ty as StyleEditionDefault>::style_edition_default(
                                style_edition
                            );
                            let mut default_str = format!("{}", default_value);
                            if default_str.is_empty() {
                                default_str = String::from("\"\"");
                            }
                            writeln!(out,
                                    "{}{} Default: {}{}",
                                    name_out,
                                    <<$ty as StyleEditionDefault>::ConfigType>::doc_hint(),
                                    default_str,
                                    if !$stb { " (unstable)" } else { "" }).unwrap();
                            $(
                                writeln!(out, "{}{}", space_str, $dstring).unwrap();
                            )+
                            writeln!(out).unwrap();
                        }
                    }
                )+
            }

            fn set_width_heuristics(&mut self, heuristics: WidthHeuristics) {
                let max_width = self.max_width.2;
                let get_width_value = |
                    was_set: bool,
                    override_value: usize,
                    heuristic_value: usize,
                    config_key: &str,
                | -> usize {
                    if !was_set {
                        return heuristic_value;
                    }
                    if override_value > max_width {
                        eprintln!(
                            "`{0}` cannot have a value that exceeds `max_width`. \
                            `{0}` will be set to the same value as `max_width`",
                            config_key,
                        );
                        return max_width;
                    }
                    override_value
                };

                let fn_call_width = get_width_value(
                    self.was_set().fn_call_width(),
                    self.fn_call_width.2,
                    heuristics.fn_call_width,
                    "fn_call_width",
                );
                self.fn_call_width.2 = fn_call_width;

                let attr_fn_like_width = get_width_value(
                    self.was_set().attr_fn_like_width(),
                    self.attr_fn_like_width.2,
                    heuristics.attr_fn_like_width,
                    "attr_fn_like_width",
                );
                self.attr_fn_like_width.2 = attr_fn_like_width;

                let struct_lit_width = get_width_value(
                    self.was_set().struct_lit_width(),
                    self.struct_lit_width.2,
                    heuristics.struct_lit_width,
                    "struct_lit_width",
                );
                self.struct_lit_width.2 = struct_lit_width;

                let struct_variant_width = get_width_value(
                    self.was_set().struct_variant_width(),
                    self.struct_variant_width.2,
                    heuristics.struct_variant_width,
                    "struct_variant_width",
                );
                self.struct_variant_width.2 = struct_variant_width;

                let array_width = get_width_value(
                    self.was_set().array_width(),
                    self.array_width.2,
                    heuristics.array_width,
                    "array_width",
                );
                self.array_width.2 = array_width;

                let chain_width = get_width_value(
                    self.was_set().chain_width(),
                    self.chain_width.2,
                    heuristics.chain_width,
                    "chain_width",
                );
                self.chain_width.2 = chain_width;

                let single_line_if_else_max_width = get_width_value(
                    self.was_set().single_line_if_else_max_width(),
                    self.single_line_if_else_max_width.2,
                    heuristics.single_line_if_else_max_width,
                    "single_line_if_else_max_width",
                );
                self.single_line_if_else_max_width.2 = single_line_if_else_max_width;

                let single_line_let_else_max_width = get_width_value(
                    self.was_set().single_line_let_else_max_width(),
                    self.single_line_let_else_max_width.2,
                    heuristics.single_line_let_else_max_width,
                    "single_line_let_else_max_width",
                );
                self.single_line_let_else_max_width.2 = single_line_let_else_max_width;
            }

            fn set_heuristics(&mut self) {
                let max_width = self.max_width.2;
                match self.use_small_heuristics.2 {
                    Heuristics::Default =>
                        self.set_width_heuristics(WidthHeuristics::scaled(max_width)),
                    Heuristics::Max => self.set_width_heuristics(WidthHeuristics::set(max_width)),
                    Heuristics::Off => self.set_width_heuristics(WidthHeuristics::null()),
                };
            }

            fn set_ignore(&mut self, dir: &Path) {
                self.ignore.2.add_prefix(dir);
            }

            fn set_merge_imports(&mut self) {
                if self.was_set().merge_imports() {
                    eprintln!(
                        "Warning: the `merge_imports` option is deprecated. \
                        Use `imports_granularity=\"Crate\"` instead"
                    );
                    if !self.was_set().imports_granularity() {
                        self.imports_granularity.2 = if self.merge_imports() {
                            ImportGranularity::Crate
                        } else {
                            ImportGranularity::Preserve
                        };
                    }
                }
            }

            fn set_fn_args_layout(&mut self) {
                if self.was_set().fn_args_layout() {
                    eprintln!(
                        "Warning: the `fn_args_layout` option is deprecated. \
                        Use `fn_params_layout`. instead"
                    );
                    if !self.was_set().fn_params_layout() {
                        self.fn_params_layout.2 = self.fn_args_layout();
                    }
                }
            }

            fn set_hide_parse_errors(&mut self) {
                if self.was_set().hide_parse_errors() {
                    eprintln!(
                        "Warning: the `hide_parse_errors` option is deprecated. \
                        Use `show_parse_errors` instead"
                    );
                    if !self.was_set().show_parse_errors() {
                        self.show_parse_errors.2 = self.hide_parse_errors();
                    }
                }
            }

            fn set_version(&mut self) {
                if !self.was_set().version() {
                    return;
                }

                eprintln!(
                    "Warning: the `version` option is deprecated. \
                    Use `style_edition` instead."
                );

                if self.was_set().style_edition() || self.was_set_cli().style_edition() {
                    eprintln!(
                        "Warning: the deprecated `version` option was \
                        used in conjunction with the `style_edition` \
                        option which takes precedence. \
                        The value of the `version` option will be ignored."
                    );
                }
            }

            #[allow(unreachable_pub)]
            /// Returns `true` if the config key was explicitly set and is the default value.
            pub fn is_default(&self, key: &str) -> bool {
                let style_edition = StyleEdition::Edition2015;
                $(
                    let default_value = <$ty as StyleEditionDefault>::style_edition_default(
                        style_edition
                    );
                    if let stringify!($i) = key {
                        return self.$i.1 && self.$i.2 == default_value;
                    }
                 )+
                false
            }
        }

        // Template for the default configuration
        impl Default for Config {
            fn default() -> Config {
                Config::default_with_style_edition(StyleEdition::Edition2015)
            }
        }
    )
}

pub(crate) fn is_stable_option_and_value<T>(
    option_name: &str,
    option_stable: bool,
    option_value: &T,
) -> bool
where
    T: PartialEq + std::fmt::Debug + ConfigType,
{
    let nightly = crate::is_nightly_channel!();
    let variant_stable = option_value.stable_variant();
    match (nightly, option_stable, variant_stable) {
        // Stable with an unstable option
        (false, false, _) => {
            eprintln!(
                "Warning: can't set `{option_name} = {option_value:?}`, unstable features are only \
                       available in nightly channel."
            );
            false
        }
        // Stable with a stable option, but an unstable variant
        (false, true, false) => {
            eprintln!(
                "Warning: can't set `{option_name} = {option_value:?}`, unstable variants are only \
                       available in nightly channel."
            );
            false
        }
        // Nightly: everything allowed
        // Stable with stable option and variant: allowed
        (true, _, _) | (false, true, true) => true,
    }
}
