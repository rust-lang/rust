use crate::config::file_lines::FileLines;
use crate::config::options::{IgnoreList, WidthHeuristics};

/// Trait for types that can be used in `Config`.
pub(crate) trait ConfigType: Sized {
    /// Returns hint text for use in `Config::print_docs()`. For enum types, this is a
    /// pipe-separated list of variants; for other types it returns "<type>".
    fn doc_hint() -> String;
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
    ($($i:ident: $ty:ty, $def:expr, $stb:expr, $( $dstring:expr ),+ );+ $(;)*) => (
        #[cfg(test)]
        use std::collections::HashSet;
        use std::io::Write;

        use serde::{Deserialize, Serialize};

        #[derive(Clone)]
        #[allow(unreachable_pub)]
        pub struct Config {
            // if a license_template_path has been specified, successfully read, parsed and compiled
            // into a regex, it will be stored here
            pub license_template: Option<Regex>,
            // For each config item, we store a bool indicating whether it has
            // been accessed and the value, and a bool whether the option was
            // manually initialised, or taken from the default,
            $($i: (Cell<bool>, bool, $ty, bool)),+
        }

        // Just like the Config struct but with each property wrapped
        // as Option<T>. This is used to parse a rustfmt.toml that doesn't
        // specify all properties of `Config`.
        // We first parse into `PartialConfig`, then create a default `Config`
        // and overwrite the properties with corresponding values from `PartialConfig`.
        #[derive(Deserialize, Serialize, Clone)]
        #[allow(unreachable_pub)]
        pub struct PartialConfig {
            $(pub $i: Option<$ty>),+
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
            pub fn $i(&mut self, value: $ty) {
                (self.0).$i.2 = value;
                match stringify!($i) {
                    "max_width"
                    | "use_small_heuristics"
                    | "fn_call_width"
                    | "single_line_if_else_max_width"
                    | "attr_fn_like_width"
                    | "struct_lit_width"
                    | "struct_variant_width"
                    | "array_width"
                    | "chain_width" => self.0.set_heuristics(),
                    "license_template_path" => self.0.set_license_template(),
                    "merge_imports" => self.0.set_merge_imports(),
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

        impl Config {
            $(
            #[allow(unreachable_pub)]
            pub fn $i(&self) -> $ty {
                self.$i.0.set(true);
                self.$i.2.clone()
            }
            )+

            #[allow(unreachable_pub)]
            pub fn set(&mut self) -> ConfigSetter<'_> {
                ConfigSetter(self)
            }

            #[allow(unreachable_pub)]
            pub fn was_set(&self) -> ConfigWasSet<'_> {
                ConfigWasSet(self)
            }

            fn fill_from_parsed_config(mut self, parsed: PartialConfig, dir: &Path) -> Config {
            $(
                if let Some(val) = parsed.$i {
                    if self.$i.3 {
                        self.$i.1 = true;
                        self.$i.2 = val;
                    } else {
                        if crate::is_nightly_channel!() {
                            self.$i.1 = true;
                            self.$i.2 = val;
                        } else {
                            eprintln!("Warning: can't set `{} = {:?}`, unstable features are only \
                                       available in nightly channel.", stringify!($i), val);
                        }
                    }
                }
            )+
                self.set_heuristics();
                self.set_license_template();
                self.set_ignore(dir);
                self.set_merge_imports();
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
                        stringify!($i) => val.parse::<$ty>().is_ok(),
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
                            self.$i.1 = true;
                            self.$i.2 = val.parse::<$ty>()
                                .expect(&format!("Failed to parse override for {} (\"{}\") as a {}",
                                                 stringify!($i),
                                                 val,
                                                 stringify!($ty)));
                        }
                    )+
                    _ => panic!("Unknown config key in override: {}", key)
                }

                match key {
                    "max_width"
                    | "use_small_heuristics"
                    | "fn_call_width"
                    | "single_line_if_else_max_width"
                    | "attr_fn_like_width"
                    | "struct_lit_width"
                    | "struct_variant_width"
                    | "array_width"
                    | "chain_width" => self.set_heuristics(),
                    "license_template_path" => self.set_license_template(),
                    "merge_imports" => self.set_merge_imports(),
                    &_ => (),
                }
            }

            #[allow(unreachable_pub)]
            pub fn is_hidden_option(name: &str) -> bool {
                const HIDE_OPTIONS: [&str; 5] =
                    ["verbose", "verbose_diff", "file_lines", "width_heuristics", "merge_imports"];
                HIDE_OPTIONS.contains(&name)
            }

            #[allow(unreachable_pub)]
            pub fn print_docs(out: &mut dyn Write, include_unstable: bool) {
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
                            let mut default_str = format!("{}", $def);
                            if default_str.is_empty() {
                                default_str = String::from("\"\"");
                            }
                            writeln!(out,
                                    "{}{} Default: {}{}",
                                    name_out,
                                    <$ty>::doc_hint(),
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

            fn set_license_template(&mut self) {
                if self.was_set().license_template_path() {
                    let lt_path = self.license_template_path();
                    if lt_path.len() > 0 {
                        match license::load_and_compile_template(&lt_path) {
                            Ok(re) => self.license_template = Some(re),
                            Err(msg) => eprintln!("Warning for license template file {:?}: {}",
                                                lt_path, msg),
                        }
                    } else {
                        self.license_template = None;
                    }
                }
            }

            fn set_ignore(&mut self, dir: &Path) {
                self.ignore.2.add_prefix(dir);
            }

            fn set_merge_imports(&mut self) {
                if self.was_set().merge_imports() {
                    eprintln!(
                        "Warning: the `merge_imports` option is deprecated. \
                        Use `imports_granularity=Crate` instead"
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

            #[allow(unreachable_pub)]
            /// Returns `true` if the config key was explicitly set and is the default value.
            pub fn is_default(&self, key: &str) -> bool {
                $(
                    if let stringify!($i) = key {
                        return self.$i.1 && self.$i.2 == $def;
                    }
                 )+
                false
            }
        }

        // Template for the default configuration
        impl Default for Config {
            fn default() -> Config {
                Config {
                    license_template: None,
                    $(
                        $i: (Cell::new(false), false, $def, $stb),
                    )+
                }
            }
        }
    )
}
