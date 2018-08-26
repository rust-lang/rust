// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use config::file_lines::FileLines;
use config::options::{IgnoreList, WidthHeuristics};

/// Trait for types that can be used in `Config`.
pub trait ConfigType: Sized {
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

/// Check if we're in a nightly build.
///
/// The environment variable `CFG_RELEASE_CHANNEL` is set during the rustc bootstrap
/// to "stable", "beta", or "nightly" depending on what toolchain is being built.
/// If we are being built as part of the stable or beta toolchains, we want
/// to disable unstable configuration options.
///
/// If we're being built by cargo (e.g. `cargo +nightly install rustfmt-nightly`),
/// `CFG_RELEASE_CHANNEL` is not set. As we only support being built against the
/// nightly compiler when installed from crates.io, default to nightly mode.
macro_rules! is_nightly_channel {
    () => {
        option_env!("CFG_RELEASE_CHANNEL")
            .map(|c| c == "nightly" || c == "dev")
            .unwrap_or(true)
    };
}

macro_rules! create_config {
    ($($i:ident: $ty:ty, $def:expr, $stb:expr, $( $dstring:expr ),+ );+ $(;)*) => (
        #[cfg(test)]
        use std::collections::HashSet;
        use std::io::Write;

        #[derive(Clone)]
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
        pub struct PartialConfig {
            $(pub $i: Option<$ty>),+
        }

        impl PartialConfig {
            pub fn to_toml(&self) -> Result<String, String> {
                // Non-user-facing options can't be specified in TOML
                let mut cloned = self.clone();
                cloned.file_lines = None;
                cloned.verbose = None;
                cloned.width_heuristics = None;

                ::toml::to_string(&cloned)
                    .map_err(|e| format!("Could not output config: {}", e.to_string()))
            }
        }

        // Macro hygiene won't allow us to make `set_$i()` methods on Config
        // for each item, so this struct is used to give the API to set values:
        // `config.set().option(false)`. It's pretty ugly. Consider replacing
        // with `config.set_option(false)` if we ever get a stable/usable
        // `concat_idents!()`.
        pub struct ConfigSetter<'a>(&'a mut Config);

        impl<'a> ConfigSetter<'a> {
            $(
            pub fn $i(&mut self, value: $ty) {
                (self.0).$i.2 = value;
                match stringify!($i) {
                    "max_width" | "use_small_heuristics" => self.0.set_heuristics(),
                    "license_template_path" => self.0.set_license_template(),
                    &_ => (),
                }
            }
            )+
        }

        // Query each option, returns true if the user set the option, false if
        // a default was used.
        pub struct ConfigWasSet<'a>(&'a Config);

        impl<'a> ConfigWasSet<'a> {
            $(
            pub fn $i(&self) -> bool {
                (self.0).$i.1
            }
            )+
        }

        impl Config {
            pub(crate) fn version_meets_requirement(&self) -> bool {
                if self.was_set().required_version() {
                    let version = env!("CARGO_PKG_VERSION");
                    let required_version = self.required_version();
                    if version != required_version {
                        println!(
                            "Error: rustfmt version ({}) doesn't match the required version ({})",
                            version,
                            required_version,
                        );
                        return false;
                    }
                }

                true
            }

            $(
            pub fn $i(&self) -> $ty {
                self.$i.0.set(true);
                self.$i.2.clone()
            }
            )+

            pub fn set<'a>(&'a mut self) -> ConfigSetter<'a> {
                ConfigSetter(self)
            }

            pub fn was_set<'a>(&'a self) -> ConfigWasSet<'a> {
                ConfigWasSet(self)
            }

            fn fill_from_parsed_config(mut self, parsed: PartialConfig, dir: &Path) -> Config {
            $(
                if let Some(val) = parsed.$i {
                    if self.$i.3 {
                        self.$i.1 = true;
                        self.$i.2 = val;
                    } else {
                        if is_nightly_channel!() {
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

            pub(crate) fn from_toml(toml: &str, dir: &Path) -> Result<Config, String> {
                let parsed: ::toml::Value =
                    toml.parse().map_err(|e| format!("Could not parse TOML: {}", e))?;
                let mut err: String = String::new();
                {
                    let table = parsed
                        .as_table()
                        .ok_or(String::from("Parsed config was not table"))?;
                    for key in table.keys() {
                        if !Config::is_valid_name(key) {
                            let msg = &format!("Warning: Unknown configuration option `{}`\n", key);
                            err.push_str(msg)
                        }
                    }
                }
                match parsed.try_into() {
                    Ok(parsed_config) => {
                        if !err.is_empty() {
                            eprint!("{}", err);
                        }
                        Ok(Config::default().fill_from_parsed_config(parsed_config, dir: &Path))
                    }
                    Err(e) => {
                        err.push_str("Error: Decoding config file failed:\n");
                        err.push_str(format!("{}\n", e).as_str());
                        err.push_str("Please check your config file.");
                        Err(err)
                    }
                }
            }

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

            pub fn all_options(&self) -> PartialConfig {
                PartialConfig {
                    $(
                        $i: Some(self.$i.2.clone()),
                    )+
                }
            }

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
                    "max_width" | "use_small_heuristics" => self.set_heuristics(),
                    "license_template_path" => self.set_license_template(),
                    &_ => (),
                }
            }

            /// Construct a `Config` from the toml file specified at `file_path`.
            ///
            /// This method only looks at the provided path, for a method that
            /// searches parents for a `rustfmt.toml` see `from_resolved_toml_path`.
            ///
            /// Return a `Config` if the config could be read and parsed from
            /// the file, Error otherwise.
            pub(super) fn from_toml_path(file_path: &Path) -> Result<Config, Error> {
                let mut file = File::open(&file_path)?;
                let mut toml = String::new();
                file.read_to_string(&mut toml)?;
                Config::from_toml(&toml, file_path.parent().unwrap())
                    .map_err(|err| Error::new(ErrorKind::InvalidData, err))
            }

            /// Resolve the config for input in `dir`.
            ///
            /// Searches for `rustfmt.toml` beginning with `dir`, and
            /// recursively checking parents of `dir` if no config file is found.
            /// If no config file exists in `dir` or in any parent, a
            /// default `Config` will be returned (and the returned path will be empty).
            ///
            /// Returns the `Config` to use, and the path of the project file if there was
            /// one.
            pub(super) fn from_resolved_toml_path(
                dir: &Path,
            ) -> Result<(Config, Option<PathBuf>), Error> {
                /// Try to find a project file in the given directory and its parents.
                /// Returns the path of a the nearest project file if one exists,
                /// or `None` if no project file was found.
                fn resolve_project_file(dir: &Path) -> Result<Option<PathBuf>, Error> {
                    let mut current = if dir.is_relative() {
                        env::current_dir()?.join(dir)
                    } else {
                        dir.to_path_buf()
                    };

                    current = fs::canonicalize(current)?;

                    loop {
                        match get_toml_path(&current) {
                            Ok(Some(path)) => return Ok(Some(path)),
                            Err(e) => return Err(e),
                            _ => ()
                        }

                        // If the current directory has no parent, we're done searching.
                        if !current.pop() {
                            return Ok(None);
                        }
                    }
                }

                match resolve_project_file(dir)? {
                    None => Ok((Config::default(), None)),
                    Some(path) => Config::from_toml_path(&path).map(|config| (config, Some(path))),
                }
            }

            pub fn is_hidden_option(name: &str) -> bool {
                const HIDE_OPTIONS: [&str; 4] =
                    ["verbose", "verbose_diff", "file_lines", "width_heuristics"];
                HIDE_OPTIONS.contains(&name)
            }

            pub fn print_docs(out: &mut Write, include_unstable: bool) {
                use std::cmp;
                let max = 0;
                $( let max = cmp::max(max, stringify!($i).len()+1); )+
                let mut space_str = String::with_capacity(max);
                for _ in 0..max {
                    space_str.push(' ');
                }
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
                            writeln!(out,
                                    "{}{} Default: {:?}{}",
                                    name_out,
                                    <$ty>::doc_hint(),
                                    $def,
                                    if !$stb { " (unstable)" } else { "" }).unwrap();
                            $(
                                writeln!(out, "{}{}", space_str, $dstring).unwrap();
                            )+
                            writeln!(out).unwrap();
                        }
                    }
                )+
            }

            fn set_heuristics(&mut self) {
                if self.use_small_heuristics.2 == Heuristics::Default {
                    let max_width = self.max_width.2;
                    self.set().width_heuristics(WidthHeuristics::scaled(max_width));
                } else if self.use_small_heuristics.2 == Heuristics::Max {
                    let max_width = self.max_width.2;
                    self.set().width_heuristics(WidthHeuristics::set(max_width));
                } else {
                    self.set().width_heuristics(WidthHeuristics::null());
                }
            }

            fn set_license_template(&mut self) {
                if self.was_set().license_template_path() {
                    let lt_path = self.license_template_path();
                    match license::load_and_compile_template(&lt_path) {
                        Ok(re) => self.license_template = Some(re),
                        Err(msg) => eprintln!("Warning for license template file {:?}: {}",
                                              lt_path, msg),
                    }
                }
            }

            fn set_ignore(&mut self, dir: &Path) {
                self.ignore.2.add_prefix(dir);
            }

            /// Returns true if the config key was explicitely set and is the default value.
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
