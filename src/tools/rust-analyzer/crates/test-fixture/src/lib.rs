//! A set of high-level utility fixture methods to use in tests.
use std::{iter, mem, ops::Not, str::FromStr, sync};

use base_db::{
    CrateDisplayName, CrateGraph, CrateId, CrateName, CrateOrigin, Dependency, Edition, Env,
    FileChange, FileSet, LangCrateOrigin, SourceDatabaseExt, SourceRoot, Version, VfsPath,
};
use cfg::CfgOptions;
use hir_expand::{
    change::ChangeWithProcMacros,
    db::ExpandDatabase,
    proc_macro::{
        ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind, ProcMacros,
    },
};
use rustc_hash::FxHashMap;
use span::{FileId, FilePosition, FileRange, Span};
use test_utils::{
    extract_range_or_offset, Fixture, FixtureWithProjectMeta, RangeOrOffset, CURSOR_MARKER,
    ESCAPED_CURSOR_MARKER,
};
use tt::{Leaf, Subtree, TokenTree};

pub const WORKSPACE: base_db::SourceRootId = base_db::SourceRootId(0);

pub trait WithFixture: Default + ExpandDatabase + SourceDatabaseExt + 'static {
    #[track_caller]
    fn with_single_file(ra_fixture: &str) -> (Self, FileId) {
        let fixture = ChangeFixture::parse(ra_fixture);
        let mut db = Self::default();
        fixture.change.apply(&mut db);
        assert_eq!(fixture.files.len(), 1, "Multiple file found in the fixture");
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

impl<DB: ExpandDatabase + SourceDatabaseExt + Default + 'static> WithFixture for DB {}

pub struct ChangeFixture {
    pub file_position: Option<(FileId, RangeOrOffset)>,
    pub files: Vec<FileId>,
    pub change: ChangeWithProcMacros,
}

const SOURCE_ROOT_PREFIX: &str = "/";

impl ChangeFixture {
    pub fn parse(ra_fixture: &str) -> ChangeFixture {
        Self::parse_with_proc_macros(ra_fixture, Vec::new())
    }

    pub fn parse_with_proc_macros(
        ra_fixture: &str,
        mut proc_macro_defs: Vec<(String, ProcMacro)>,
    ) -> ChangeFixture {
        let FixtureWithProjectMeta {
            fixture,
            mini_core,
            proc_macro_names,
            toolchain,
            target_data_layout,
        } = FixtureWithProjectMeta::parse(ra_fixture);
        let target_data_layout = Ok(target_data_layout.into());
        let toolchain = Some({
            let channel = toolchain.as_deref().unwrap_or("stable");
            Version::parse(&format!("1.76.0-{channel}")).unwrap()
        });
        let mut source_change = FileChange::new();

        let mut files = Vec::new();
        let mut crate_graph = CrateGraph::default();
        let mut crates = FxHashMap::default();
        let mut crate_deps = Vec::new();
        let mut default_crate_root: Option<FileId> = None;
        let mut default_cfg = CfgOptions::default();
        let mut default_env = Env::new_for_test_fixture();

        let mut file_set = FileSet::default();
        let mut current_source_root_kind = SourceRootKind::Local;
        let mut file_id = FileId::from_raw(0);
        let mut roots = Vec::new();

        let mut file_position = None;

        for entry in fixture {
            let text = if entry.text.contains(CURSOR_MARKER) {
                if entry.text.contains(ESCAPED_CURSOR_MARKER) {
                    entry.text.replace(ESCAPED_CURSOR_MARKER, CURSOR_MARKER).into()
                } else {
                    let (range_or_offset, text) = extract_range_or_offset(&entry.text);
                    assert!(file_position.is_none());
                    file_position = Some((file_id, range_or_offset));
                    text.into()
                }
            } else {
                entry.text.as_str().into()
            };

            let meta = FileMeta::from_fixture(entry, current_source_root_kind);
            assert!(meta.path.starts_with(SOURCE_ROOT_PREFIX));
            if !meta.deps.is_empty() {
                assert!(meta.krate.is_some(), "can't specify deps without naming the crate")
            }

            if let Some(kind) = meta.introduce_new_source_root {
                assert!(
                    meta.krate.is_some(),
                    "new_source_root meta doesn't make sense without crate meta"
                );
                let prev_kind = mem::replace(&mut current_source_root_kind, kind);
                let prev_root = match prev_kind {
                    SourceRootKind::Local => SourceRoot::new_local(mem::take(&mut file_set)),
                    SourceRootKind::Library => SourceRoot::new_library(mem::take(&mut file_set)),
                };
                roots.push(prev_root);
            }

            if let Some((krate, origin, version)) = meta.krate {
                let crate_name = CrateName::normalize_dashes(&krate);
                let crate_id = crate_graph.add_crate_root(
                    file_id,
                    meta.edition,
                    Some(crate_name.clone().into()),
                    version,
                    meta.cfg.clone(),
                    Some(meta.cfg),
                    meta.env,
                    false,
                    origin,
                );
                let prev = crates.insert(crate_name.clone(), crate_id);
                assert!(prev.is_none(), "multiple crates with same name: {}", crate_name);
                for dep in meta.deps {
                    let prelude = match &meta.extern_prelude {
                        Some(v) => v.contains(&dep),
                        None => true,
                    };
                    let dep = CrateName::normalize_dashes(&dep);
                    crate_deps.push((crate_name.clone(), dep, prelude))
                }
            } else if meta.path == "/main.rs" || meta.path == "/lib.rs" {
                assert!(default_crate_root.is_none());
                default_crate_root = Some(file_id);
                default_cfg.extend(meta.cfg.into_iter());
                default_env.extend(meta.env.iter().map(|(x, y)| (x.to_owned(), y.to_owned())));
            }

            source_change.change_file(file_id, Some(text));
            let path = VfsPath::new_virtual_path(meta.path);
            file_set.insert(file_id, path);
            files.push(file_id);
            file_id = FileId::from_raw(file_id.index() + 1);
        }

        if crates.is_empty() {
            let crate_root = default_crate_root
                .expect("missing default crate root, specify a main.rs or lib.rs");
            crate_graph.add_crate_root(
                crate_root,
                Edition::CURRENT,
                Some(CrateName::new("test").unwrap().into()),
                None,
                default_cfg.clone(),
                Some(default_cfg),
                default_env,
                false,
                CrateOrigin::Local { repo: None, name: None },
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

        if let Some(mini_core) = mini_core {
            let core_file = file_id;
            file_id = FileId::from_raw(file_id.index() + 1);

            let mut fs = FileSet::default();
            fs.insert(core_file, VfsPath::new_virtual_path("/sysroot/core/lib.rs".to_owned()));
            roots.push(SourceRoot::new_library(fs));

            source_change.change_file(core_file, Some(mini_core.source_code().into()));

            let all_crates = crate_graph.crates_in_topological_order();

            let core_crate = crate_graph.add_crate_root(
                core_file,
                Edition::Edition2021,
                Some(CrateDisplayName::from_canonical_name("core".to_owned())),
                None,
                Default::default(),
                Default::default(),
                Env::new_for_test_fixture(),
                false,
                CrateOrigin::Lang(LangCrateOrigin::Core),
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

            proc_macro_defs.extend(default_test_proc_macros());
            let (proc_macro, source) = filter_test_proc_macros(&proc_macro_names, proc_macro_defs);
            let mut fs = FileSet::default();
            fs.insert(
                proc_lib_file,
                VfsPath::new_virtual_path("/sysroot/proc_macros/lib.rs".to_owned()),
            );
            roots.push(SourceRoot::new_library(fs));

            source_change.change_file(proc_lib_file, Some(source.into()));

            let all_crates = crate_graph.crates_in_topological_order();

            let proc_macros_crate = crate_graph.add_crate_root(
                proc_lib_file,
                Edition::Edition2021,
                Some(CrateDisplayName::from_canonical_name("proc_macros".to_owned())),
                None,
                Default::default(),
                Default::default(),
                Env::new_for_test_fixture(),
                true,
                CrateOrigin::Local { repo: None, name: None },
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

        let mut change = ChangeWithProcMacros {
            source_change,
            proc_macros: proc_macros.is_empty().not().then_some(proc_macros),
            toolchains: Some(iter::repeat(toolchain).take(crate_graph.len()).collect()),
            target_data_layouts: Some(
                iter::repeat(target_data_layout).take(crate_graph.len()).collect(),
            ),
        };

        change.source_change.set_roots(roots);
        change.source_change.set_crate_graph(crate_graph);

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
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(IdentityProcMacroExpander),
                disabled: false,
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
                kind: ProcMacroKind::CustomDerive,
                expander: sync::Arc::new(IdentityProcMacroExpander),
                disabled: false,
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
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(AttributeInputReplaceProcMacroExpander),
                disabled: false,
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
                kind: ProcMacroKind::FuncLike,
                expander: sync::Arc::new(MirrorProcMacroExpander),
                disabled: false,
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
                kind: ProcMacroKind::FuncLike,
                expander: sync::Arc::new(ShortenProcMacroExpander),
                disabled: false,
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
    extern_prelude: Option<Vec<String>>,
    cfg: CfgOptions,
    edition: Edition,
    env: Env,
    introduce_new_source_root: Option<SourceRootKind>,
}

impl FileMeta {
    fn from_fixture(f: Fixture, current_source_root_kind: SourceRootKind) -> Self {
        let mut cfg = CfgOptions::default();
        for (k, v) in f.cfgs {
            if let Some(v) = v {
                cfg.insert_key_value(k.into(), v.into());
            } else {
                cfg.insert_atom(k.into());
            }
        }

        let introduce_new_source_root = f.introduce_new_source_root.map(|kind| match &*kind {
            "local" => SourceRootKind::Local,
            "library" => SourceRootKind::Library,
            invalid => panic!("invalid source root kind '{invalid}'"),
        });
        let current_source_root_kind =
            introduce_new_source_root.unwrap_or(current_source_root_kind);

        let deps = f.deps;
        Self {
            path: f.path,
            krate: f.krate.map(|it| parse_crate(it, current_source_root_kind, f.library)),
            extern_prelude: f.extern_prelude,
            deps,
            cfg,
            edition: f.edition.map_or(Edition::CURRENT, |v| Edition::from_str(&v).unwrap()),
            env: f.env.into_iter().collect(),
            introduce_new_source_root,
        }
    }
}

fn parse_crate(
    crate_str: String,
    current_source_root_kind: SourceRootKind,
    explicit_non_workspace_member: bool,
) -> (String, CrateOrigin, Option<String>) {
    // syntax:
    //   "my_awesome_crate"
    //   "my_awesome_crate@0.0.1,http://example.com"
    let (name, repo, version) = if let Some((name, remain)) = crate_str.split_once('@') {
        let (version, repo) =
            remain.split_once(',').expect("crate meta: found '@' without version and url");
        (name.to_owned(), Some(repo.to_owned()), Some(version.to_owned()))
    } else {
        (crate_str, None, None)
    };

    let non_workspace_member = explicit_non_workspace_member
        || matches!(current_source_root_kind, SourceRootKind::Library);

    let origin = match LangCrateOrigin::from(&*name) {
        LangCrateOrigin::Other => {
            let name = name.clone();
            if non_workspace_member {
                CrateOrigin::Library { repo, name }
            } else {
                CrateOrigin::Local { repo, name: Some(name) }
            }
        }
        origin => CrateOrigin::Lang(origin),
    };

    (name, origin, version)
}

// Identity mapping
#[derive(Debug)]
struct IdentityProcMacroExpander;
impl ProcMacroExpander for IdentityProcMacroExpander {
    fn expand(
        &self,
        subtree: &Subtree<Span>,
        _: Option<&Subtree<Span>>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
    ) -> Result<Subtree<Span>, ProcMacroExpansionError> {
        Ok(subtree.clone())
    }
}

// Pastes the attribute input as its output
#[derive(Debug)]
struct AttributeInputReplaceProcMacroExpander;
impl ProcMacroExpander for AttributeInputReplaceProcMacroExpander {
    fn expand(
        &self,
        _: &Subtree<Span>,
        attrs: Option<&Subtree<Span>>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
    ) -> Result<Subtree<Span>, ProcMacroExpansionError> {
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
        input: &Subtree<Span>,
        _: Option<&Subtree<Span>>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
    ) -> Result<Subtree<Span>, ProcMacroExpansionError> {
        fn traverse(input: &Subtree<Span>) -> Subtree<Span> {
            let mut token_trees = vec![];
            for tt in input.token_trees.iter().rev() {
                let tt = match tt {
                    tt::TokenTree::Leaf(leaf) => tt::TokenTree::Leaf(leaf.clone()),
                    tt::TokenTree::Subtree(sub) => tt::TokenTree::Subtree(traverse(sub)),
                };
                token_trees.push(tt);
            }
            Subtree { delimiter: input.delimiter, token_trees: token_trees.into_boxed_slice() }
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
        input: &Subtree<Span>,
        _: Option<&Subtree<Span>>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
    ) -> Result<Subtree<Span>, ProcMacroExpansionError> {
        return Ok(traverse(input));

        fn traverse(input: &Subtree<Span>) -> Subtree<Span> {
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

        fn modify_leaf(leaf: &Leaf<Span>) -> Leaf<Span> {
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
