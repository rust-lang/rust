//! A set of high-level utility fixture methods to use in tests.
use std::{mem, str::FromStr, sync};

use cfg::CfgOptions;
use rustc_hash::FxHashMap;
use test_utils::{
    extract_range_or_offset, Fixture, FixtureWithProjectMeta, RangeOrOffset, CURSOR_MARKER,
    ESCAPED_CURSOR_MARKER,
};
use triomphe::Arc;
use tt::token_id::{Leaf, Subtree, TokenTree};
use vfs::{file_set::FileSet, VfsPath};

use crate::{
    input::{CrateName, CrateOrigin, LangCrateOrigin},
    Change, CrateDisplayName, CrateGraph, CrateId, Dependency, Edition, Env, FileId, FilePosition,
    FileRange, ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacros, ReleaseChannel,
    SourceDatabaseExt, SourceRoot, SourceRootId,
};

pub const WORKSPACE: SourceRootId = SourceRootId(0);

pub trait WithFixture: Default + SourceDatabaseExt + 'static {
    #[track_caller]
    fn with_single_file(ra_fixture: &str) -> (Self, FileId) {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert_eq!(fixture.files.len(), 1);
        (db, fixture.files[0])
    }

    #[track_caller]
    fn with_many_files(ra_fixture: &str) -> (Self, Vec<FileId>) {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        (db, fixture.files)
    }

    #[track_caller]
    fn with_files(ra_fixture: &str) -> Self {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        db
    }

    #[track_caller]
    fn with_files_extra_proc_macros(
        ra_fixture: &str,
        proc_macros: Vec<(String, ProcMacro)>,
    ) -> Self {
        let fixture = ChangeFixture::parse_with_proc_macros(ra_fixture, proc_macros);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        db
    }

    #[track_caller]
    fn with_position(ra_fixture: &str) -> (Self, FilePosition) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let offset = range_or_offset.expect_offset();
        (db, FilePosition { file_id, offset })
    }

    #[track_caller]
    fn with_range(ra_fixture: &str) -> (Self, FileRange) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let range = range_or_offset.expect_range();
        (db, FileRange { file_id, range })
    }

    #[track_caller]
    fn with_range_or_offset(ra_fixture: &str) -> (Self, FileId, RangeOrOffset) {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        let (file_id, range_or_offset) = fixture
            .file_position
            .expect("Could not find file position in fixture. Did you forget to add an `$0`?");
        (db, file_id, range_or_offset)
    }

    fn test_crate(&self) -> CrateId {
        let crate_graph = self.crate_graph();
        let mut it = crate_graph.iter();
        let res = it.next().unwrap();
        assert!(it.next().is_none());
        res
    }
}

impl<DB: SourceDatabaseExt + Default + 'static> WithFixture for DB {}

pub struct ChangeFixture {
    pub file_position: Option<(FileId, RangeOrOffset)>,
    pub files: Vec<FileId>,
    pub change: Change,
}

impl ChangeFixture {
    pub fn parse(ra_fixture: &str) -> ChangeFixture {
        Self::parse_with_proc_macros(ra_fixture, Vec::new())
    }

    pub fn parse_with_proc_macros(
        ra_fixture: &str,
        mut proc_macro_defs: Vec<(String, ProcMacro)>,
    ) -> ChangeFixture {
        let FixtureWithProjectMeta { fixture, mini_core, proc_macro_names, toolchain } =
            FixtureWithProjectMeta::parse(ra_fixture);
        let toolchain = toolchain
            .map(|it| {
                ReleaseChannel::from_str(&it)
                    .unwrap_or_else(|| panic!("unknown release channel found: {it}"))
            })
            .unwrap_or(ReleaseChannel::Stable);
        let mut change = Change::new();

        let mut files = Vec::new();
        let mut crate_graph = CrateGraph::default();
        let mut crates = FxHashMap::default();
        let mut crate_deps = Vec::new();
        let mut default_crate_root: Option<FileId> = None;
        let mut default_target_data_layout: Option<String> = None;
        let mut default_cfg = CfgOptions::default();

        let mut file_set = FileSet::default();
        let mut current_source_root_kind = SourceRootKind::Local;
        let source_root_prefix = "/".to_string();
        let mut file_id = FileId(0);
        let mut roots = Vec::new();

        let mut file_position = None;

        for entry in fixture {
            let text = if entry.text.contains(CURSOR_MARKER) {
                if entry.text.contains(ESCAPED_CURSOR_MARKER) {
                    entry.text.replace(ESCAPED_CURSOR_MARKER, CURSOR_MARKER)
                } else {
                    let (range_or_offset, text) = extract_range_or_offset(&entry.text);
                    assert!(file_position.is_none());
                    file_position = Some((file_id, range_or_offset));
                    text
                }
            } else {
                entry.text.clone()
            };

            let meta = FileMeta::from(entry);
            assert!(meta.path.starts_with(&source_root_prefix));
            if !meta.deps.is_empty() {
                assert!(meta.krate.is_some(), "can't specify deps without naming the crate")
            }

            if let Some(kind) = &meta.introduce_new_source_root {
                let root = match current_source_root_kind {
                    SourceRootKind::Local => SourceRoot::new_local(mem::take(&mut file_set)),
                    SourceRootKind::Library => SourceRoot::new_library(mem::take(&mut file_set)),
                };
                roots.push(root);
                current_source_root_kind = *kind;
            }

            if let Some((krate, origin, version)) = meta.krate {
                let crate_name = CrateName::normalize_dashes(&krate);
                let crate_id = crate_graph.add_crate_root(
                    file_id,
                    meta.edition,
                    Some(crate_name.clone().into()),
                    version,
                    meta.cfg,
                    Default::default(),
                    meta.env,
                    false,
                    origin,
                    meta.target_data_layout
                        .as_deref()
                        .map(Arc::from)
                        .ok_or_else(|| "target_data_layout unset".into()),
                    Some(toolchain),
                );
                let prev = crates.insert(crate_name.clone(), crate_id);
                assert!(prev.is_none());
                for dep in meta.deps {
                    let prelude = meta.extern_prelude.contains(&dep);
                    let dep = CrateName::normalize_dashes(&dep);
                    crate_deps.push((crate_name.clone(), dep, prelude))
                }
            } else if meta.path == "/main.rs" || meta.path == "/lib.rs" {
                assert!(default_crate_root.is_none());
                default_crate_root = Some(file_id);
                default_cfg = meta.cfg;
                default_target_data_layout = meta.target_data_layout;
            }

            change.change_file(file_id, Some(Arc::from(text)));
            let path = VfsPath::new_virtual_path(meta.path);
            file_set.insert(file_id, path);
            files.push(file_id);
            file_id.0 += 1;
        }

        if crates.is_empty() {
            let crate_root = default_crate_root
                .expect("missing default crate root, specify a main.rs or lib.rs");
            crate_graph.add_crate_root(
                crate_root,
                Edition::CURRENT,
                Some(CrateName::new("test").unwrap().into()),
                None,
                default_cfg,
                Default::default(),
                Env::new_for_test_fixture(),
                false,
                CrateOrigin::Local { repo: None, name: None },
                default_target_data_layout
                    .map(|x| x.into())
                    .ok_or_else(|| "target_data_layout unset".into()),
                Some(toolchain),
            );
        } else {
            for (from, to, prelude) in crate_deps {
                let from_id = crates[&from];
                let to_id = crates[&to];
                crate_graph
                    .add_dep(
                        from_id,
                        Dependency::with_prelude(CrateName::new(&to).unwrap(), to_id, prelude),
                    )
                    .unwrap();
            }
        }
        let target_layout = crate_graph.iter().next().map_or_else(
            || Err("target_data_layout unset".into()),
            |it| crate_graph[it].target_layout.clone(),
        );

        if let Some(mini_core) = mini_core {
            let core_file = file_id;
            file_id.0 += 1;

            let mut fs = FileSet::default();
            fs.insert(core_file, VfsPath::new_virtual_path("/sysroot/core/lib.rs".to_string()));
            roots.push(SourceRoot::new_library(fs));

            change.change_file(core_file, Some(Arc::from(mini_core.source_code())));

            let all_crates = crate_graph.crates_in_topological_order();

            let core_crate = crate_graph.add_crate_root(
                core_file,
                Edition::Edition2021,
                Some(CrateDisplayName::from_canonical_name("core".to_string())),
                None,
                Default::default(),
                Default::default(),
                Env::new_for_test_fixture(),
                false,
                CrateOrigin::Lang(LangCrateOrigin::Core),
                target_layout.clone(),
                Some(toolchain),
            );

            for krate in all_crates {
                crate_graph
                    .add_dep(krate, Dependency::new(CrateName::new("core").unwrap(), core_crate))
                    .unwrap();
            }
        }

        let mut proc_macros = ProcMacros::default();
        if !proc_macro_names.is_empty() {
            let proc_lib_file = file_id;
            file_id.0 += 1;

            proc_macro_defs.extend(default_test_proc_macros());
            let (proc_macro, source) = filter_test_proc_macros(&proc_macro_names, proc_macro_defs);
            let mut fs = FileSet::default();
            fs.insert(
                proc_lib_file,
                VfsPath::new_virtual_path("/sysroot/proc_macros/lib.rs".to_string()),
            );
            roots.push(SourceRoot::new_library(fs));

            change.change_file(proc_lib_file, Some(Arc::from(source)));

            let all_crates = crate_graph.crates_in_topological_order();

            let proc_macros_crate = crate_graph.add_crate_root(
                proc_lib_file,
                Edition::Edition2021,
                Some(CrateDisplayName::from_canonical_name("proc_macros".to_string())),
                None,
                Default::default(),
                Default::default(),
                Env::new_for_test_fixture(),
                true,
                CrateOrigin::Local { repo: None, name: None },
                target_layout,
                Some(toolchain),
            );
            proc_macros.insert(proc_macros_crate, Ok(proc_macro));

            for krate in all_crates {
                crate_graph
                    .add_dep(
                        krate,
                        Dependency::new(CrateName::new("proc_macros").unwrap(), proc_macros_crate),
                    )
                    .unwrap();
            }
        }

        let root = match current_source_root_kind {
            SourceRootKind::Local => SourceRoot::new_local(mem::take(&mut file_set)),
            SourceRootKind::Library => SourceRoot::new_library(mem::take(&mut file_set)),
        };
        roots.push(root);
        change.set_roots(roots);
        change.set_crate_graph(crate_graph);
        change.set_proc_macros(proc_macros);

        ChangeFixture { file_position, files, change }
    }
}

fn default_test_proc_macros() -> [(String, ProcMacro); 5] {
    [
        (
            r#"
#[proc_macro_attribute]
pub fn identity(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
"#
            .into(),
            ProcMacro {
                name: "identity".into(),
                kind: crate::ProcMacroKind::Attr,
                expander: sync::Arc::new(IdentityProcMacroExpander),
            },
        ),
        (
            r#"
#[proc_macro_derive(DeriveIdentity)]
pub fn derive_identity(item: TokenStream) -> TokenStream {
    item
}
"#
            .into(),
            ProcMacro {
                name: "DeriveIdentity".into(),
                kind: crate::ProcMacroKind::CustomDerive,
                expander: sync::Arc::new(IdentityProcMacroExpander),
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn input_replace(attr: TokenStream, _item: TokenStream) -> TokenStream {
    attr
}
"#
            .into(),
            ProcMacro {
                name: "input_replace".into(),
                kind: crate::ProcMacroKind::Attr,
                expander: sync::Arc::new(AttributeInputReplaceProcMacroExpander),
            },
        ),
        (
            r#"
#[proc_macro]
pub fn mirror(input: TokenStream) -> TokenStream {
    input
}
"#
            .into(),
            ProcMacro {
                name: "mirror".into(),
                kind: crate::ProcMacroKind::FuncLike,
                expander: sync::Arc::new(MirrorProcMacroExpander),
            },
        ),
        (
            r#"
#[proc_macro]
pub fn shorten(input: TokenStream) -> TokenStream {
    loop {}
}
"#
            .into(),
            ProcMacro {
                name: "shorten".into(),
                kind: crate::ProcMacroKind::FuncLike,
                expander: sync::Arc::new(ShortenProcMacroExpander),
            },
        ),
    ]
}

fn filter_test_proc_macros(
    proc_macro_names: &[String],
    proc_macro_defs: Vec<(String, ProcMacro)>,
) -> (Vec<ProcMacro>, String) {
    // The source here is only required so that paths to the macros exist and are resolvable.
    let mut source = String::new();
    let mut proc_macros = Vec::new();

    for (c, p) in proc_macro_defs {
        if !proc_macro_names.iter().any(|name| name == &stdx::to_lower_snake_case(&p.name)) {
            continue;
        }
        proc_macros.push(p);
        source += &c;
    }

    (proc_macros, source)
}

#[derive(Debug, Clone, Copy)]
enum SourceRootKind {
    Local,
    Library,
}

#[derive(Debug)]
struct FileMeta {
    path: String,
    krate: Option<(String, CrateOrigin, Option<String>)>,
    deps: Vec<String>,
    extern_prelude: Vec<String>,
    cfg: CfgOptions,
    edition: Edition,
    env: Env,
    introduce_new_source_root: Option<SourceRootKind>,
    target_data_layout: Option<String>,
}

fn parse_crate(crate_str: String) -> (String, CrateOrigin, Option<String>) {
    if let Some((a, b)) = crate_str.split_once('@') {
        let (version, origin) = match b.split_once(':') {
            Some(("CratesIo", data)) => match data.split_once(',') {
                Some((version, url)) => {
                    (version, CrateOrigin::Local { repo: Some(url.to_owned()), name: None })
                }
                _ => panic!("Bad crates.io parameter: {data}"),
            },
            _ => panic!("Bad string for crate origin: {b}"),
        };
        (a.to_owned(), origin, Some(version.to_string()))
    } else {
        let crate_origin = match LangCrateOrigin::from(&*crate_str) {
            LangCrateOrigin::Other => CrateOrigin::Local { repo: None, name: None },
            origin => CrateOrigin::Lang(origin),
        };
        (crate_str, crate_origin, None)
    }
}

impl From<Fixture> for FileMeta {
    fn from(f: Fixture) -> FileMeta {
        let mut cfg = CfgOptions::default();
        f.cfg_atoms.iter().for_each(|it| cfg.insert_atom(it.into()));
        f.cfg_key_values.iter().for_each(|(k, v)| cfg.insert_key_value(k.into(), v.into()));
        let deps = f.deps;
        FileMeta {
            path: f.path,
            krate: f.krate.map(parse_crate),
            extern_prelude: f.extern_prelude.unwrap_or_else(|| deps.clone()),
            deps,
            cfg,
            edition: f.edition.as_ref().map_or(Edition::CURRENT, |v| Edition::from_str(v).unwrap()),
            env: f.env.into_iter().collect(),
            introduce_new_source_root: f.introduce_new_source_root.map(|kind| match &*kind {
                "local" => SourceRootKind::Local,
                "library" => SourceRootKind::Library,
                invalid => panic!("invalid source root kind '{invalid}'"),
            }),
            target_data_layout: f.target_data_layout,
        }
    }
}

// Identity mapping
#[derive(Debug)]
struct IdentityProcMacroExpander;
impl ProcMacroExpander for IdentityProcMacroExpander {
    fn expand(
        &self,
        subtree: &Subtree,
        _: Option<&Subtree>,
        _: &Env,
    ) -> Result<Subtree, ProcMacroExpansionError> {
        Ok(subtree.clone())
    }
}

// Pastes the attribute input as its output
#[derive(Debug)]
struct AttributeInputReplaceProcMacroExpander;
impl ProcMacroExpander for AttributeInputReplaceProcMacroExpander {
    fn expand(
        &self,
        _: &Subtree,
        attrs: Option<&Subtree>,
        _: &Env,
    ) -> Result<Subtree, ProcMacroExpansionError> {
        attrs
            .cloned()
            .ok_or_else(|| ProcMacroExpansionError::Panic("Expected attribute input".into()))
    }
}

#[derive(Debug)]
struct MirrorProcMacroExpander;
impl ProcMacroExpander for MirrorProcMacroExpander {
    fn expand(
        &self,
        input: &Subtree,
        _: Option<&Subtree>,
        _: &Env,
    ) -> Result<Subtree, ProcMacroExpansionError> {
        fn traverse(input: &Subtree) -> Subtree {
            let mut token_trees = vec![];
            for tt in input.token_trees.iter().rev() {
                let tt = match tt {
                    tt::TokenTree::Leaf(leaf) => tt::TokenTree::Leaf(leaf.clone()),
                    tt::TokenTree::Subtree(sub) => tt::TokenTree::Subtree(traverse(sub)),
                };
                token_trees.push(tt);
            }
            Subtree { delimiter: input.delimiter, token_trees }
        }
        Ok(traverse(input))
    }
}

// Replaces every literal with an empty string literal and every identifier with its first letter,
// but retains all tokens' span. Useful for testing we don't assume token hasn't been modified by
// macros even if it retains its span.
#[derive(Debug)]
struct ShortenProcMacroExpander;
impl ProcMacroExpander for ShortenProcMacroExpander {
    fn expand(
        &self,
        input: &Subtree,
        _: Option<&Subtree>,
        _: &Env,
    ) -> Result<Subtree, ProcMacroExpansionError> {
        return Ok(traverse(input));

        fn traverse(input: &Subtree) -> Subtree {
            let token_trees = input
                .token_trees
                .iter()
                .map(|it| match it {
                    TokenTree::Leaf(leaf) => tt::TokenTree::Leaf(modify_leaf(leaf)),
                    TokenTree::Subtree(subtree) => tt::TokenTree::Subtree(traverse(subtree)),
                })
                .collect();
            Subtree { delimiter: input.delimiter, token_trees }
        }

        fn modify_leaf(leaf: &Leaf) -> Leaf {
            let mut leaf = leaf.clone();
            match &mut leaf {
                Leaf::Literal(it) => {
                    // XXX Currently replaces any literals with an empty string, but supporting
                    // "shortening" other literals would be nice.
                    it.text = "\"\"".into();
                }
                Leaf::Punct(_) => {}
                Leaf::Ident(it) => {
                    it.text = it.text.chars().take(1).collect();
                }
            }
            leaf
        }
    }
}
