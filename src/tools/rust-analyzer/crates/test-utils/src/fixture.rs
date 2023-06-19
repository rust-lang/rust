//! Defines `Fixture` -- a convenient way to describe the initial state of
//! rust-analyzer database from a single string.
//!
//! Fixtures are strings containing rust source code with optional metadata.
//! A fixture without metadata is parsed into a single source file.
//! Use this to test functionality local to one file.
//!
//! Simple Example:
//! ```
//! r#"
//! fn main() {
//!     println!("Hello World")
//! }
//! "#
//! ```
//!
//! Metadata can be added to a fixture after a `//-` comment.
//! The basic form is specifying filenames,
//! which is also how to define multiple files in a single test fixture
//!
//! Example using two files in the same crate:
//! ```
//! "
//! //- /main.rs
//! mod foo;
//! fn main() {
//!     foo::bar();
//! }
//!
//! //- /foo.rs
//! pub fn bar() {}
//! "
//! ```
//!
//! Example using two crates with one file each, with one crate depending on the other:
//! ```
//! r#"
//! //- /main.rs crate:a deps:b
//! fn main() {
//!     b::foo();
//! }
//! //- /lib.rs crate:b
//! pub fn b() {
//!     println!("Hello World")
//! }
//! "#
//! ```
//!
//! Metadata allows specifying all settings and variables
//! that are available in a real rust project:
//! - crate names via `crate:cratename`
//! - dependencies via `deps:dep1,dep2`
//! - configuration settings via `cfg:dbg=false,opt_level=2`
//! - environment variables via `env:PATH=/bin,RUST_LOG=debug`
//!
//! Example using all available metadata:
//! ```
//! "
//! //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
//! fn insert_source_code_here() {}
//! "
//! ```

use std::iter;

use rustc_hash::FxHashMap;
use stdx::trim_indent;

#[derive(Debug, Eq, PartialEq)]
pub struct Fixture {
    pub path: String,
    pub text: String,
    pub krate: Option<String>,
    pub deps: Vec<String>,
    pub extern_prelude: Option<Vec<String>>,
    pub cfg_atoms: Vec<String>,
    pub cfg_key_values: Vec<(String, String)>,
    pub edition: Option<String>,
    pub env: FxHashMap<String, String>,
    pub introduce_new_source_root: Option<String>,
    pub target_data_layout: Option<String>,
}

pub struct MiniCore {
    activated_flags: Vec<String>,
    valid_flags: Vec<String>,
}

pub struct FixtureWithProjectMeta {
    pub fixture: Vec<Fixture>,
    pub mini_core: Option<MiniCore>,
    pub proc_macro_names: Vec<String>,
    pub toolchain: Option<String>,
}

impl FixtureWithProjectMeta {
    /// Parses text which looks like this:
    ///
    ///  ```not_rust
    ///  //- some meta
    ///  line 1
    ///  line 2
    ///  //- other meta
    ///  ```
    ///
    /// Fixture can also start with a proc_macros and minicore declaration (in that order):
    ///
    /// ```
    /// //- toolchain: nightly
    /// //- proc_macros: identity
    /// //- minicore: sized
    /// ```
    ///
    /// That will set toolchain to nightly and include predefined proc macros and a subset of
    /// `libcore` into the fixture, see `minicore.rs` for what's available. Note that toolchain
    /// defaults to stable.
    pub fn parse(ra_fixture: &str) -> Self {
        let fixture = trim_indent(ra_fixture);
        let mut fixture = fixture.as_str();
        let mut toolchain = None;
        let mut mini_core = None;
        let mut res: Vec<Fixture> = Vec::new();
        let mut proc_macro_names = vec![];

        if let Some(meta) = fixture.strip_prefix("//- toolchain:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            toolchain = Some(meta.trim().to_string());
            fixture = remain;
        }

        if let Some(meta) = fixture.strip_prefix("//- proc_macros:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            proc_macro_names = meta.split(',').map(|it| it.trim().to_string()).collect();
            fixture = remain;
        }

        if let Some(meta) = fixture.strip_prefix("//- minicore:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            mini_core = Some(MiniCore::parse(meta));
            fixture = remain;
        }

        let default = if fixture.contains("//-") { None } else { Some("//- /main.rs") };

        for (ix, line) in default.into_iter().chain(fixture.split_inclusive('\n')).enumerate() {
            if line.contains("//-") {
                assert!(
                    line.starts_with("//-"),
                    "Metadata line {ix} has invalid indentation. \
                     All metadata lines need to have the same indentation.\n\
                     The offending line: {line:?}"
                );
            }

            if line.starts_with("//-") {
                let meta = Self::parse_meta_line(line);
                res.push(meta);
            } else {
                if line.starts_with("// ")
                    && line.contains(':')
                    && !line.contains("::")
                    && !line.contains('.')
                    && line.chars().all(|it| !it.is_uppercase())
                {
                    panic!("looks like invalid metadata line: {line:?}");
                }

                if let Some(entry) = res.last_mut() {
                    entry.text.push_str(line);
                }
            }
        }

        Self { fixture: res, mini_core, proc_macro_names, toolchain }
    }

    //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
    fn parse_meta_line(meta: &str) -> Fixture {
        assert!(meta.starts_with("//-"));
        let meta = meta["//-".len()..].trim();
        let components = meta.split_ascii_whitespace().collect::<Vec<_>>();

        let path = components[0].to_string();
        assert!(path.starts_with('/'), "fixture path does not start with `/`: {path:?}");

        let mut krate = None;
        let mut deps = Vec::new();
        let mut extern_prelude = None;
        let mut edition = None;
        let mut cfg_atoms = Vec::new();
        let mut cfg_key_values = Vec::new();
        let mut env = FxHashMap::default();
        let mut introduce_new_source_root = None;
        let mut target_data_layout = Some(
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".to_string(),
        );
        for component in components[1..].iter() {
            let (key, value) =
                component.split_once(':').unwrap_or_else(|| panic!("invalid meta line: {meta:?}"));
            match key {
                "crate" => krate = Some(value.to_string()),
                "deps" => deps = value.split(',').map(|it| it.to_string()).collect(),
                "extern-prelude" => {
                    if value.is_empty() {
                        extern_prelude = Some(Vec::new());
                    } else {
                        extern_prelude =
                            Some(value.split(',').map(|it| it.to_string()).collect::<Vec<_>>());
                    }
                }
                "edition" => edition = Some(value.to_string()),
                "cfg" => {
                    for entry in value.split(',') {
                        match entry.split_once('=') {
                            Some((k, v)) => cfg_key_values.push((k.to_string(), v.to_string())),
                            None => cfg_atoms.push(entry.to_string()),
                        }
                    }
                }
                "env" => {
                    for key in value.split(',') {
                        if let Some((k, v)) = key.split_once('=') {
                            env.insert(k.into(), v.into());
                        }
                    }
                }
                "new_source_root" => introduce_new_source_root = Some(value.to_string()),
                "target_data_layout" => target_data_layout = Some(value.to_string()),
                _ => panic!("bad component: {component:?}"),
            }
        }

        for prelude_dep in extern_prelude.iter().flatten() {
            assert!(
                deps.contains(prelude_dep),
                "extern-prelude {extern_prelude:?} must be a subset of deps {deps:?}"
            );
        }

        Fixture {
            path,
            text: String::new(),
            krate,
            deps,
            extern_prelude,
            cfg_atoms,
            cfg_key_values,
            edition,
            env,
            introduce_new_source_root,
            target_data_layout,
        }
    }
}

impl MiniCore {
    const RAW_SOURCE: &str = include_str!("./minicore.rs");

    fn has_flag(&self, flag: &str) -> bool {
        self.activated_flags.iter().any(|it| it == flag)
    }

    pub fn from_flags<'a>(flags: impl IntoIterator<Item = &'a str>) -> Self {
        MiniCore {
            activated_flags: flags.into_iter().map(|x| x.to_owned()).collect(),
            valid_flags: Vec::new(),
        }
    }

    #[track_caller]
    fn assert_valid_flag(&self, flag: &str) {
        if !self.valid_flags.iter().any(|it| it == flag) {
            panic!("invalid flag: {flag:?}, valid flags: {:?}", self.valid_flags);
        }
    }

    fn parse(line: &str) -> MiniCore {
        let mut res = MiniCore { activated_flags: Vec::new(), valid_flags: Vec::new() };

        for entry in line.trim().split(", ") {
            if res.has_flag(entry) {
                panic!("duplicate minicore flag: {entry:?}");
            }
            res.activated_flags.push(entry.to_owned());
        }

        res
    }

    pub fn available_flags() -> impl Iterator<Item = &'static str> {
        let lines = MiniCore::RAW_SOURCE.split_inclusive('\n');
        lines
            .map_while(|x| x.strip_prefix("//!"))
            .skip_while(|line| !line.contains("Available flags:"))
            .skip(1)
            .map(|x| x.split_once(':').unwrap().0.trim())
    }

    /// Strips parts of minicore.rs which are flagged by inactive flags.
    ///
    /// This is probably over-engineered to support flags dependencies.
    pub fn source_code(mut self) -> String {
        let mut buf = String::new();
        let mut lines = MiniCore::RAW_SOURCE.split_inclusive('\n');

        let mut implications = Vec::new();

        // Parse `//!` preamble and extract flags and dependencies.
        let trim_doc: fn(&str) -> Option<&str> = |line| match line.strip_prefix("//!") {
            Some(it) => Some(it),
            None => {
                assert!(line.trim().is_empty(), "expected empty line after minicore header");
                None
            }
        };
        for line in lines
            .by_ref()
            .map_while(trim_doc)
            .skip_while(|line| !line.contains("Available flags:"))
            .skip(1)
        {
            let (flag, deps) = line.split_once(':').unwrap();
            let flag = flag.trim();

            self.valid_flags.push(flag.to_string());
            implications.extend(
                iter::repeat(flag)
                    .zip(deps.split(", ").map(str::trim).filter(|dep| !dep.is_empty())),
            );
        }

        for (_, dep) in &implications {
            self.assert_valid_flag(dep);
        }

        for flag in &self.activated_flags {
            self.assert_valid_flag(flag);
        }

        // Fixed point loop to compute transitive closure of flags.
        loop {
            let mut changed = false;
            for &(u, v) in &implications {
                if self.has_flag(u) && !self.has_flag(v) {
                    self.activated_flags.push(v.to_string());
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let mut active_regions = Vec::new();
        let mut seen_regions = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if let Some(region) = trimmed.strip_prefix("// region:") {
                active_regions.push(region);
                continue;
            }
            if let Some(region) = trimmed.strip_prefix("// endregion:") {
                let prev = active_regions.pop().unwrap();
                assert_eq!(prev, region, "unbalanced region pairs");
                continue;
            }

            let mut line_region = false;
            if let Some(idx) = trimmed.find("// :") {
                line_region = true;
                active_regions.push(&trimmed[idx + "// :".len()..]);
            }

            let mut keep = true;
            for &region in &active_regions {
                assert!(!region.starts_with(' '), "region marker starts with a space: {region:?}");
                self.assert_valid_flag(region);
                seen_regions.push(region);
                keep &= self.has_flag(region);
            }

            if keep {
                buf.push_str(line);
            }
            if line_region {
                active_regions.pop().unwrap();
            }
        }

        if !active_regions.is_empty() {
            panic!("unclosed regions: {:?} Add an `endregion` comment", active_regions);
        }

        for flag in &self.valid_flags {
            if !seen_regions.iter().any(|it| it == flag) {
                panic!("unused minicore flag: {flag:?}");
            }
        }
        buf
    }
}

#[test]
#[should_panic]
fn parse_fixture_checks_further_indented_metadata() {
    FixtureWithProjectMeta::parse(
        r"
        //- /lib.rs
          mod bar;

          fn foo() {}
          //- /bar.rs
          pub fn baz() {}
          ",
    );
}

#[test]
fn parse_fixture_gets_full_meta() {
    let FixtureWithProjectMeta { fixture: parsed, mini_core, proc_macro_names, toolchain } =
        FixtureWithProjectMeta::parse(
            r#"
//- toolchain: nightly
//- proc_macros: identity
//- minicore: coerce_unsized
//- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b,atom env:OUTDIR=path/to,OTHER=foo
mod m;
"#,
        );
    assert_eq!(toolchain, Some("nightly".to_string()));
    assert_eq!(proc_macro_names, vec!["identity".to_string()]);
    assert_eq!(mini_core.unwrap().activated_flags, vec!["coerce_unsized".to_string()]);
    assert_eq!(1, parsed.len());

    let meta = &parsed[0];
    assert_eq!("mod m;\n", meta.text);

    assert_eq!("foo", meta.krate.as_ref().unwrap());
    assert_eq!("/lib.rs", meta.path);
    assert_eq!(2, meta.env.len());
}
