//! A set of high-level utility fixture methods to use in tests.
use std::{any::TypeId, mem, str::FromStr, sync};

use base_db::{
    Crate, CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CrateWorkspaceData,
    DependencyBuilder, Env, FileChange, FileSet, LangCrateOrigin, SourceDatabase, SourceRoot,
    Version, VfsPath, salsa,
};
use cfg::CfgOptions;
use hir_expand::{
    EditionedFileId, FileRange,
    change::ChangeWithProcMacros,
    db::ExpandDatabase,
    files::FilePosition,
    proc_macro::{
        ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind, ProcMacrosBuilder,
    },
    quote,
    tt::{Leaf, TokenTree, TopSubtree, TopSubtreeBuilder, TtElement, TtIter},
};
use intern::{Symbol, sym};
use paths::AbsPathBuf;
use rustc_hash::FxHashMap;
use span::{Edition, FileId, Span};
use stdx::itertools::Itertools;
use test_utils::{
    CURSOR_MARKER, ESCAPED_CURSOR_MARKER, Fixture, FixtureWithProjectMeta, RangeOrOffset,
    extract_range_or_offset,
};
use triomphe::Arc;

pub const WORKSPACE: base_db::SourceRootId = base_db::SourceRootId(0);

pub trait WithFixture: Default + ExpandDatabase + SourceDatabase + 'static {
    #[track_caller]
    fn with_single_file(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (Self, EditionedFileId) {
        let mut db = Self::default();
        let fixture = ChangeFixture::parse(&db, ra_fixture);
        fixture.change.apply(&mut db);
        assert_eq!(fixture.files.len(), 1, "Multiple file found in the fixture");
        (db, fixture.files[0])
    }

    #[track_caller]
    fn with_many_files(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (Self, Vec<EditionedFileId>) {
        let mut db = Self::default();
        let fixture = ChangeFixture::parse(&db, ra_fixture);
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        (db, fixture.files)
    }

    #[track_caller]
    fn with_files(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> Self {
        let mut db = Self::default();
        let fixture = ChangeFixture::parse(&db, ra_fixture);
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        db
    }

    #[track_caller]
    fn with_files_extra_proc_macros(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        proc_macros: Vec<(String, ProcMacro)>,
    ) -> Self {
        let mut db = Self::default();
        let fixture = ChangeFixture::parse_with_proc_macros(&db, ra_fixture, proc_macros);
        fixture.change.apply(&mut db);
        assert!(fixture.file_position.is_none());
        db
    }

    #[track_caller]
    fn with_position(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (Self, FilePosition) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let offset = range_or_offset.expect_offset();
        (db, FilePosition { file_id, offset })
    }

    #[track_caller]
    fn with_range(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> (Self, FileRange) {
        let (db, file_id, range_or_offset) = Self::with_range_or_offset(ra_fixture);
        let range = range_or_offset.expect_range();
        (db, FileRange { file_id, range })
    }

    #[track_caller]
    fn with_range_or_offset(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> (Self, EditionedFileId, RangeOrOffset) {
        let mut db = Self::default();
        let fixture = ChangeFixture::parse(&db, ra_fixture);
        fixture.change.apply(&mut db);

        let (file_id, range_or_offset) = fixture
            .file_position
            .expect("Could not find file position in fixture. Did you forget to add an `$0`?");
        (db, file_id, range_or_offset)
    }

    fn test_crate(&self) -> Crate {
        self.all_crates().iter().copied().find(|&krate| !krate.data(self).origin.is_lang()).unwrap()
    }
}

impl<DB: ExpandDatabase + SourceDatabase + Default + 'static> WithFixture for DB {}

pub struct ChangeFixture {
    pub file_position: Option<(EditionedFileId, RangeOrOffset)>,
    pub files: Vec<EditionedFileId>,
    pub change: ChangeWithProcMacros,
}

const SOURCE_ROOT_PREFIX: &str = "/";

impl ChangeFixture {
    pub fn parse(
        db: &dyn salsa::Database,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
    ) -> ChangeFixture {
        Self::parse_with_proc_macros(db, ra_fixture, Vec::new())
    }

    pub fn parse_with_proc_macros(
        db: &dyn salsa::Database,
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
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
        let mut source_change = FileChange::default();

        let mut files = Vec::new();
        let mut crate_graph = CrateGraphBuilder::default();
        let mut crates = FxHashMap::default();
        let mut crate_deps = Vec::new();
        let mut default_crate_root: Option<FileId> = None;
        let mut default_edition = Edition::CURRENT;
        let mut default_cfg = CfgOptions::default();
        let mut default_env = Env::from_iter([(
            String::from("__ra_is_test_fixture"),
            String::from("__ra_is_test_fixture"),
        )]);

        let mut file_set = FileSet::default();
        let mut current_source_root_kind = SourceRootKind::Local;
        let mut file_id = FileId::from_raw(0);
        let mut roots = Vec::new();

        let mut file_position = None;

        let crate_ws_data =
            Arc::new(CrateWorkspaceData { data_layout: target_data_layout, toolchain });

        // FIXME: This is less than ideal
        let proc_macro_cwd = Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap()));

        for entry in fixture {
            let mut range_or_offset = None;
            let text = if entry.text.contains(CURSOR_MARKER) {
                if entry.text.contains(ESCAPED_CURSOR_MARKER) {
                    entry.text.replace(ESCAPED_CURSOR_MARKER, CURSOR_MARKER)
                } else {
                    let (roo, text) = extract_range_or_offset(&entry.text);
                    assert!(file_position.is_none());
                    range_or_offset = Some(roo);
                    text
                }
            } else {
                entry.text.as_str().into()
            };

            let meta = FileMeta::from_fixture(entry, current_source_root_kind);
            if let Some(range_or_offset) = range_or_offset {
                file_position =
                    Some((EditionedFileId::new(db, file_id, meta.edition), range_or_offset));
            }

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
                    origin,
                    false,
                    proc_macro_cwd.clone(),
                    crate_ws_data.clone(),
                );
                let prev = crates.insert(crate_name.clone(), crate_id);
                assert!(prev.is_none(), "multiple crates with same name: {crate_name}");
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
                default_edition = meta.edition;
                default_cfg.extend(meta.cfg.into_iter());
                default_env.extend_from_other(&meta.env);
            }

            source_change.change_file(file_id, Some(text));
            let path = VfsPath::new_virtual_path(meta.path);
            file_set.insert(file_id, path);
            files.push(EditionedFileId::new(db, file_id, meta.edition));
            file_id = FileId::from_raw(file_id.index() + 1);
        }

        if crates.is_empty() {
            let crate_root = default_crate_root
                .expect("missing default crate root, specify a main.rs or lib.rs");
            crate_graph.add_crate_root(
                crate_root,
                default_edition,
                Some(CrateName::new("ra_test_fixture").unwrap().into()),
                None,
                default_cfg.clone(),
                Some(default_cfg),
                default_env,
                CrateOrigin::Local { repo: None, name: None },
                false,
                proc_macro_cwd.clone(),
                crate_ws_data.clone(),
            );
        } else {
            for (from, to, prelude) in crate_deps {
                let from_id = crates[&from];
                let to_id = crates[&to];
                let sysroot = crate_graph[to_id].basic.origin.is_lang();
                crate_graph
                    .add_dep(
                        from_id,
                        DependencyBuilder::with_prelude(to.clone(), to_id, prelude, sysroot),
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

            source_change.change_file(core_file, Some(mini_core.source_code()));

            let all_crates = crate_graph.iter().collect::<Vec<_>>();

            let core_crate = crate_graph.add_crate_root(
                core_file,
                Edition::CURRENT,
                Some(CrateDisplayName::from_canonical_name("core")),
                None,
                Default::default(),
                Default::default(),
                Env::from_iter([(
                    String::from("__ra_is_test_fixture"),
                    String::from("__ra_is_test_fixture"),
                )]),
                CrateOrigin::Lang(LangCrateOrigin::Core),
                false,
                proc_macro_cwd.clone(),
                crate_ws_data.clone(),
            );

            for krate in all_crates {
                crate_graph
                    .add_dep(
                        krate,
                        DependencyBuilder::with_prelude(
                            CrateName::new("core").unwrap(),
                            core_crate,
                            true,
                            true,
                        ),
                    )
                    .unwrap();
            }
        }

        let mut proc_macros = ProcMacrosBuilder::default();
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

            source_change.change_file(proc_lib_file, Some(source));

            let all_crates = crate_graph.iter().collect::<Vec<_>>();

            let proc_macros_crate = crate_graph.add_crate_root(
                proc_lib_file,
                Edition::CURRENT,
                Some(CrateDisplayName::from_canonical_name("proc_macros")),
                None,
                Default::default(),
                Default::default(),
                Env::from_iter([(
                    String::from("__ra_is_test_fixture"),
                    String::from("__ra_is_test_fixture"),
                )]),
                CrateOrigin::Local { repo: None, name: None },
                true,
                proc_macro_cwd,
                crate_ws_data,
            );
            proc_macros.insert(proc_macros_crate, Ok(proc_macro));

            for krate in all_crates {
                crate_graph
                    .add_dep(
                        krate,
                        DependencyBuilder::new(
                            CrateName::new("proc_macros").unwrap(),
                            proc_macros_crate,
                        ),
                    )
                    .unwrap();
            }
        }

        let root = match current_source_root_kind {
            SourceRootKind::Local => SourceRoot::new_local(mem::take(&mut file_set)),
            SourceRootKind::Library => SourceRoot::new_library(mem::take(&mut file_set)),
        };
        roots.push(root);

        let mut change = ChangeWithProcMacros { source_change, proc_macros: Some(proc_macros) };

        change.source_change.set_roots(roots);
        change.source_change.set_crate_graph(crate_graph);

        ChangeFixture { file_position, files, change }
    }
}

fn default_test_proc_macros() -> Box<[(String, ProcMacro)]> {
    Box::new([
        (
            r#"
#[proc_macro_attribute]
pub fn identity(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("identity"),
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
                name: Symbol::intern("DeriveIdentity"),
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
                name: Symbol::intern("input_replace"),
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
                name: Symbol::intern("mirror"),
                kind: ProcMacroKind::Bang,
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
                name: Symbol::intern("shorten"),
                kind: ProcMacroKind::Bang,
                expander: sync::Arc::new(ShortenProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn issue_18089(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    loop {}
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("issue_18089"),
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(Issue18089ProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn issue_18840(_attr: TokenStream, _item: TokenStream) -> TokenStream {
    loop {}
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("issue_18840"),
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(Issue18840ProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro]
pub fn issue_17479(input: TokenStream) -> TokenStream {
    input
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("issue_17479"),
                kind: ProcMacroKind::Bang,
                expander: sync::Arc::new(Issue17479ProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn issue_18898(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("issue_18898"),
                kind: ProcMacroKind::Bang,
                expander: sync::Arc::new(Issue18898ProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn disallow_cfg(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("disallow_cfg"),
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(DisallowCfgProcMacroExpander),
                disabled: false,
            },
        ),
        (
            r#"
#[proc_macro_attribute]
pub fn generate_suffixed_type(_attr: TokenStream, input: TokenStream) -> TokenStream {
    input
}
"#
            .into(),
            ProcMacro {
                name: Symbol::intern("generate_suffixed_type"),
                kind: ProcMacroKind::Attr,
                expander: sync::Arc::new(GenerateSuffixedTypeProcMacroExpander),
                disabled: false,
            },
        ),
    ])
}

fn filter_test_proc_macros(
    proc_macro_names: &[String],
    proc_macro_defs: Vec<(String, ProcMacro)>,
) -> (Vec<ProcMacro>, String) {
    // The source here is only required so that paths to the macros exist and are resolvable.
    let mut source = String::new();
    let mut proc_macros = Vec::new();

    for (c, p) in proc_macro_defs {
        if !proc_macro_names.iter().any(|name| name == &stdx::to_lower_snake_case(p.name.as_str()))
        {
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
                cfg.insert_key_value(Symbol::intern(&k), Symbol::intern(&v));
            } else {
                cfg.insert_atom(Symbol::intern(&k));
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
            let name = Symbol::intern(&name);
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
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        Ok(subtree.clone())
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Expands to a macro_rules! macro, for issue #18089.
#[derive(Debug)]
struct Issue18089ProcMacroExpander;
impl ProcMacroExpander for Issue18089ProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        call_site: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let tt::TokenTree::Leaf(macro_name) = &subtree.0[2] else {
            return Err(ProcMacroExpansionError::Panic("incorrect input".to_owned()));
        };
        Ok(quote! { call_site =>
            #[macro_export]
            macro_rules! my_macro___ {
                ($($token:tt)*) => {{
                }};
            }

            pub use my_macro___ as #macro_name;

            #subtree
        })
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Pastes the attribute input as its output
#[derive(Debug)]
struct AttributeInputReplaceProcMacroExpander;
impl ProcMacroExpander for AttributeInputReplaceProcMacroExpander {
    fn expand(
        &self,
        _: &TopSubtree,
        attrs: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        attrs
            .cloned()
            .ok_or_else(|| ProcMacroExpansionError::Panic("Expected attribute input".into()))
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

#[derive(Debug)]
struct Issue18840ProcMacroExpander;
impl ProcMacroExpander for Issue18840ProcMacroExpander {
    fn expand(
        &self,
        fn_: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        def_site: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        // Input:
        // ```
        // #[issue_18840]
        // fn foo() { let loop {} }
        // ```

        // The span that was created by the fixup infra.
        let fixed_up_span = fn_.token_trees().flat_tokens()[5].first_span();
        let mut result =
            quote! {fixed_up_span => ::core::compile_error! { "my cool compile_error!" } };
        // Make it so we won't remove the top subtree when reversing fixups.
        let top_subtree_delimiter_mut = result.top_subtree_delimiter_mut();
        top_subtree_delimiter_mut.open = def_site;
        top_subtree_delimiter_mut.close = def_site;
        Ok(result)
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

#[derive(Debug)]
struct MirrorProcMacroExpander;
impl ProcMacroExpander for MirrorProcMacroExpander {
    fn expand(
        &self,
        input: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        fn traverse(builder: &mut TopSubtreeBuilder, iter: TtIter<'_>) {
            for tt in iter.collect_vec().into_iter().rev() {
                match tt {
                    TtElement::Leaf(leaf) => builder.push(leaf.clone()),
                    TtElement::Subtree(subtree, subtree_iter) => {
                        builder.open(subtree.delimiter.kind, subtree.delimiter.open);
                        traverse(builder, subtree_iter);
                        builder.close(subtree.delimiter.close);
                    }
                }
            }
        }
        let mut builder = TopSubtreeBuilder::new(input.top_subtree().delimiter);
        traverse(&mut builder, input.iter());
        Ok(builder.build())
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
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
        input: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let mut result = input.0.clone();
        for it in &mut result {
            if let TokenTree::Leaf(leaf) = it {
                modify_leaf(leaf)
            }
        }
        return Ok(tt::TopSubtree(result));

        fn modify_leaf(leaf: &mut Leaf) {
            match leaf {
                Leaf::Literal(it) => {
                    // XXX Currently replaces any literals with an empty string, but supporting
                    // "shortening" other literals would be nice.
                    it.symbol = Symbol::empty();
                }
                Leaf::Punct(_) => {}
                Leaf::Ident(it) => {
                    it.sym = Symbol::intern(&it.sym.as_str().chars().take(1).collect::<String>());
                }
            }
        }
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Reads ident type within string quotes, for issue #17479.
#[derive(Debug)]
struct Issue17479ProcMacroExpander;
impl ProcMacroExpander for Issue17479ProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let TokenTree::Leaf(Leaf::Literal(lit)) = &subtree.0[1] else {
            return Err(ProcMacroExpansionError::Panic("incorrect Input".into()));
        };
        let symbol = &lit.symbol;
        let span = lit.span;
        Ok(quote! { span =>
            #symbol()
        })
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Reads ident type within string quotes, for issue #17479.
#[derive(Debug)]
struct Issue18898ProcMacroExpander;
impl ProcMacroExpander for Issue18898ProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        def_site: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let span = subtree
            .token_trees()
            .flat_tokens()
            .last()
            .ok_or_else(|| ProcMacroExpansionError::Panic("malformed input".to_owned()))?
            .first_span();
        let overly_long_subtree = quote! {span =>
            {
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
                let a = 5;
            }
        };
        Ok(quote! { def_site =>
            fn foo() {
                #overly_long_subtree
            }
        })
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Reads ident type within string quotes, for issue #17479.
#[derive(Debug)]
struct DisallowCfgProcMacroExpander;
impl ProcMacroExpander for DisallowCfgProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _: Option<&TopSubtree>,
        _: &Env,
        _: Span,
        _: Span,
        _: Span,
        _: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        for tt in subtree.token_trees().flat_tokens() {
            if let tt::TokenTree::Leaf(tt::Leaf::Ident(ident)) = tt
                && (ident.sym == sym::cfg || ident.sym == sym::cfg_attr)
            {
                return Err(ProcMacroExpansionError::Panic(
                    "cfg or cfg_attr found in DisallowCfgProcMacroExpander".to_owned(),
                ));
            }
        }
        Ok(subtree.clone())
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}

// Generates a new type by adding a suffix to the original name
#[derive(Debug)]
struct GenerateSuffixedTypeProcMacroExpander;
impl ProcMacroExpander for GenerateSuffixedTypeProcMacroExpander {
    fn expand(
        &self,
        subtree: &TopSubtree,
        _attrs: Option<&TopSubtree>,
        _env: &Env,
        _def_site: Span,
        call_site: Span,
        _mixed_site: Span,
        _current_dir: String,
    ) -> Result<TopSubtree, ProcMacroExpansionError> {
        let TokenTree::Leaf(Leaf::Ident(ident)) = &subtree.0[1] else {
            return Err(ProcMacroExpansionError::Panic("incorrect Input".into()));
        };

        let ident = match ident.sym.as_str() {
            "struct" => {
                let TokenTree::Leaf(Leaf::Ident(ident)) = &subtree.0[2] else {
                    return Err(ProcMacroExpansionError::Panic("incorrect Input".into()));
                };
                ident
            }

            "enum" => {
                let TokenTree::Leaf(Leaf::Ident(ident)) = &subtree.0[4] else {
                    return Err(ProcMacroExpansionError::Panic("incorrect Input".into()));
                };
                ident
            }

            _ => {
                return Err(ProcMacroExpansionError::Panic("incorrect Input".into()));
            }
        };

        let generated_ident = tt::Ident {
            sym: Symbol::intern(&format!("{}Suffix", ident.sym)),
            span: ident.span,
            is_raw: tt::IdentIsRaw::No,
        };

        let ret = quote! { call_site =>
            #subtree

            struct #generated_ident;
        };

        Ok(ret)
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        other.type_id() == TypeId::of::<Self>()
    }
}
