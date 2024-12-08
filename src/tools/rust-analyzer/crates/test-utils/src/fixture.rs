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
//! that are available in a real rust project. See [`Fixture`]
//! for the syntax.
//!
//! Example using some available metadata:
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
    /// Specifies the path for this file. It must start with "/".
    pub path: String,
    /// Defines a new crate and make this file its root module.
    ///
    /// Version and repository URL of the crate can optionally be specified; if
    /// either one is specified, the other must also be specified.
    ///
    /// Syntax:
    /// - `crate:my_awesome_lib`
    /// - `crate:my_awesome_lib@0.0.1,https://example.com/repo.git`
    pub krate: Option<String>,
    /// Specifies dependencies of this crate. This must be used with `crate` meta.
    ///
    /// Syntax: `deps:hir-def,ide-assists`
    pub deps: Vec<String>,
    /// Limits crates in the extern prelude. The set of crate names must be a
    /// subset of `deps`. This must be used with `crate` meta.
    ///
    /// If this is not specified, all the dependencies will be in the extern prelude.
    ///
    /// Syntax: `extern-prelude:hir-def,ide-assists`
    pub extern_prelude: Option<Vec<String>>,
    /// Specifies configuration options to be enabled. Options may have associated
    /// values.
    ///
    /// Syntax: `cfg:test,dbg=false,opt_level=2`
    pub cfgs: Vec<(String, Option<String>)>,
    /// Specifies the edition of this crate. This must be used with `crate` meta. If
    /// this is not specified, ([`base_db::input::Edition::CURRENT`]) will be used.
    /// This must be used with `crate` meta.
    ///
    /// Syntax: `edition:2021`
    pub edition: Option<String>,
    /// Specifies environment variables.
    ///
    /// Syntax: `env:PATH=/bin,RUST_LOG=debug`
    pub env: FxHashMap<String, String>,
    /// Introduces a new [source root](base_db::input::SourceRoot). This file **and
    /// the following files** will belong the new source root. This must be used
    /// with `crate` meta.
    ///
    /// Use this if you want to test something that uses `SourceRoot::is_library()`
    /// to check editability.
    ///
    /// Note that files before the first fixture with `new_source_root` meta will
    /// belong to an implicitly defined local source root.
    ///
    /// Syntax:
    /// - `new_source_root:library`
    /// - `new_source_root:local`
    pub introduce_new_source_root: Option<String>,
    /// Explicitly declares this crate as a library outside current workspace. This
    /// must be used with `crate` meta.
    ///
    /// This is implied if this file belongs to a library source root.
    ///
    /// Use this if you want to test something that checks if a crate is a workspace
    /// member via [`CrateOrigin`](base_db::input::CrateOrigin).
    ///
    /// Syntax: `library`
    pub library: bool,
    /// Actual file contents. All meta comments are stripped.
    pub text: String,
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
    /// Specifies LLVM data layout to be used.
    ///
    /// You probably don't want to manually specify this. See LLVM manual for the
    /// syntax, if you must: <https://llvm.org/docs/LangRef.html#data-layout>
    pub target_data_layout: String,
}

impl FixtureWithProjectMeta {
    /// Parses text which looks like this:
    ///
    ///  ```text
    ///  //- some meta
    ///  line 1
    ///  line 2
    ///  //- other meta
    ///  ```
    ///
    /// Fixture can also start with a proc_macros and minicore declaration (in that order):
    ///
    /// ```text
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
        let mut target_data_layout =
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128".to_owned();
        let mut mini_core = None;
        let mut res: Vec<Fixture> = Vec::new();
        let mut proc_macro_names = vec![];

        if let Some(meta) = fixture.strip_prefix("//- toolchain:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            toolchain = Some(meta.trim().to_owned());
            fixture = remain;
        }

        if let Some(meta) = fixture.strip_prefix("//- target_data_layout:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            meta.trim().clone_into(&mut target_data_layout);
            fixture = remain;
        }

        if let Some(meta) = fixture.strip_prefix("//- proc_macros:") {
            let (meta, remain) = meta.split_once('\n').unwrap();
            proc_macro_names = meta.split(',').map(|it| it.trim().to_owned()).collect();
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

        Self { fixture: res, mini_core, proc_macro_names, toolchain, target_data_layout }
    }

    //- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b env:OUTDIR=path/to,OTHER=foo
    fn parse_meta_line(meta: &str) -> Fixture {
        assert!(meta.starts_with("//-"));
        let meta = meta["//-".len()..].trim();
        let mut components = meta.split_ascii_whitespace();

        let path = components.next().expect("fixture meta must start with a path").to_owned();
        assert!(path.starts_with('/'), "fixture path does not start with `/`: {path:?}");

        let mut krate = None;
        let mut deps = Vec::new();
        let mut extern_prelude = None;
        let mut edition = None;
        let mut cfgs = Vec::new();
        let mut env = FxHashMap::default();
        let mut introduce_new_source_root = None;
        let mut library = false;
        for component in components {
            if component == "library" {
                library = true;
                continue;
            }

            let (key, value) =
                component.split_once(':').unwrap_or_else(|| panic!("invalid meta line: {meta:?}"));
            match key {
                "crate" => krate = Some(value.to_owned()),
                "deps" => deps = value.split(',').map(|it| it.to_owned()).collect(),
                "extern-prelude" => {
                    if value.is_empty() {
                        extern_prelude = Some(Vec::new());
                    } else {
                        extern_prelude =
                            Some(value.split(',').map(|it| it.to_owned()).collect::<Vec<_>>());
                    }
                }
                "edition" => edition = Some(value.to_owned()),
                "cfg" => {
                    for entry in value.split(',') {
                        match entry.split_once('=') {
                            Some((k, v)) => cfgs.push((k.to_owned(), Some(v.to_owned()))),
                            None => cfgs.push((entry.to_owned(), None)),
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
                "new_source_root" => introduce_new_source_root = Some(value.to_owned()),
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
            cfgs,
            edition,
            env,
            introduce_new_source_root,
            library,
        }
    }
}

impl MiniCore {
    const RAW_SOURCE: &'static str = include_str!("./minicore.rs");

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

            self.valid_flags.push(flag.to_owned());
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
                    self.activated_flags.push(v.to_owned());
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
            panic!("unclosed regions: {active_regions:?} Add an `endregion` comment");
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
    let FixtureWithProjectMeta {
        fixture: parsed,
        mini_core,
        proc_macro_names,
        toolchain,
        target_data_layout: _,
    } = FixtureWithProjectMeta::parse(
        r#"
//- toolchain: nightly
//- proc_macros: identity
//- minicore: coerce_unsized
//- /lib.rs crate:foo deps:bar,baz cfg:foo=a,bar=b,atom env:OUTDIR=path/to,OTHER=foo
mod m;
"#,
    );
    assert_eq!(toolchain, Some("nightly".to_owned()));
    assert_eq!(proc_macro_names, vec!["identity".to_owned()]);
    assert_eq!(mini_core.unwrap().activated_flags, vec!["coerce_unsized".to_owned()]);
    assert_eq!(1, parsed.len());

    let meta = &parsed[0];
    assert_eq!("mod m;\n", meta.text);

    assert_eq!("foo", meta.krate.as_ref().unwrap());
    assert_eq!("/lib.rs", meta.path);
    assert_eq!(2, meta.env.len());
}
