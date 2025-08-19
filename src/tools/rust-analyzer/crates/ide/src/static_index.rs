//! This module provides `StaticIndex` which is used for powering
//! read-only code browsers and emitting LSIF

use arrayvec::ArrayVec;
use hir::{Crate, Module, Semantics, db::HirDatabase};
use ide_db::{
    FileId, FileRange, FxHashMap, FxHashSet, RootDatabase,
    base_db::{RootQueryDb, SourceDatabase, VfsPath},
    defs::{Definition, IdentClass},
    documentation::Documentation,
    famous_defs::FamousDefs,
};
use span::Edition;
use syntax::{AstNode, SyntaxKind::*, SyntaxNode, SyntaxToken, T, TextRange};

use crate::navigation_target::UpmappingResult;
use crate::{
    Analysis, Fold, HoverConfig, HoverResult, InlayHint, InlayHintsConfig, TryToNav,
    hover::{SubstTyLen, hover_for_definition},
    inlay_hints::{AdjustmentHintsMode, InlayFieldsToResolve},
    moniker::{MonikerResult, SymbolInformationKind, def_to_kind, def_to_moniker},
    parent_module::crates_for,
};

/// A static representation of fully analyzed source code.
///
/// The intended use-case is powering read-only code browsers and emitting LSIF/SCIP.
#[derive(Debug)]
pub struct StaticIndex<'a> {
    pub files: Vec<StaticIndexedFile>,
    pub tokens: TokenStore,
    analysis: &'a Analysis,
    db: &'a RootDatabase,
    def_map: FxHashMap<Definition, TokenId>,
}

#[derive(Debug)]
pub struct ReferenceData {
    pub range: FileRange,
    pub is_definition: bool,
}

#[derive(Debug)]
pub struct TokenStaticData {
    pub documentation: Option<Documentation>,
    pub hover: Option<HoverResult>,
    pub definition: Option<FileRange>,
    pub references: Vec<ReferenceData>,
    pub moniker: Option<MonikerResult>,
    pub display_name: Option<String>,
    pub signature: Option<String>,
    pub kind: SymbolInformationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(usize);

impl TokenId {
    pub fn raw(self) -> usize {
        self.0
    }
}

#[derive(Default, Debug)]
pub struct TokenStore(Vec<TokenStaticData>);

impl TokenStore {
    pub fn insert(&mut self, data: TokenStaticData) -> TokenId {
        let id = TokenId(self.0.len());
        self.0.push(data);
        id
    }

    pub fn get_mut(&mut self, id: TokenId) -> Option<&mut TokenStaticData> {
        self.0.get_mut(id.0)
    }

    pub fn get(&self, id: TokenId) -> Option<&TokenStaticData> {
        self.0.get(id.0)
    }

    pub fn iter(self) -> impl Iterator<Item = (TokenId, TokenStaticData)> {
        self.0.into_iter().enumerate().map(|(id, data)| (TokenId(id), data))
    }
}

#[derive(Debug)]
pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,
    pub inlay_hints: Vec<InlayHint>,
    pub tokens: Vec<(TextRange, TokenId)>,
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module()).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

fn documentation_for_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: Definition,
    scope_node: &SyntaxNode,
) -> Option<Documentation> {
    let famous_defs = match &def {
        Definition::BuiltinType(_) => Some(FamousDefs(sema, sema.scope(scope_node)?.krate())),
        _ => None,
    };

    def.docs(
        sema.db,
        famous_defs.as_ref(),
        def.krate(sema.db)
            .unwrap_or_else(|| {
                (*sema.db.all_crates().last().expect("no crate graph present")).into()
            })
            .to_display_target(sema.db),
    )
}

// FIXME: This is a weird function
fn get_definitions(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> Option<ArrayVec<Definition, 2>> {
    for token in sema.descend_into_macros_exact(token) {
        let def = IdentClass::classify_token(sema, &token).map(IdentClass::definitions_no_ops);
        if let Some(defs) = def
            && !defs.is_empty()
        {
            return Some(defs);
        }
    }
    None
}

pub enum VendoredLibrariesConfig<'a> {
    Included { workspace_root: &'a VfsPath },
    Excluded,
}

impl StaticIndex<'_> {
    fn add_file(&mut self, file_id: FileId) {
        let current_crate = crates_for(self.db, file_id).pop().map(Into::into);
        let folds = self.analysis.folding_ranges(file_id).unwrap();
        let inlay_hints = self
            .analysis
            .inlay_hints(
                &InlayHintsConfig {
                    render_colons: true,
                    discriminant_hints: crate::DiscriminantHints::Fieldless,
                    type_hints: true,
                    sized_bound: false,
                    parameter_hints: true,
                    generic_parameter_hints: crate::GenericParameterHints {
                        type_hints: false,
                        lifetime_hints: false,
                        const_hints: true,
                    },
                    chaining_hints: true,
                    closure_return_type_hints: crate::ClosureReturnTypeHints::WithBlock,
                    lifetime_elision_hints: crate::LifetimeElisionHints::Never,
                    adjustment_hints: crate::AdjustmentHints::Never,
                    adjustment_hints_mode: AdjustmentHintsMode::Prefix,
                    adjustment_hints_hide_outside_unsafe: false,
                    implicit_drop_hints: false,
                    hide_named_constructor_hints: false,
                    hide_closure_initialization_hints: false,
                    hide_closure_parameter_hints: false,
                    closure_style: hir::ClosureStyle::ImplFn,
                    param_names_for_lifetime_elision_hints: false,
                    binding_mode_hints: false,
                    max_length: Some(25),
                    closure_capture_hints: false,
                    closing_brace_hints_min_lines: Some(25),
                    fields_to_resolve: InlayFieldsToResolve::empty(),
                    range_exclusive_hints: false,
                },
                file_id,
                None,
            )
            .unwrap();
        // hovers
        let sema = hir::Semantics::new(self.db);
        let root = sema.parse_guess_edition(file_id).syntax().clone();
        let edition = sema
            .attach_first_edition(file_id)
            .map(|it| it.edition(self.db))
            .unwrap_or(Edition::CURRENT);
        let display_target = match sema.first_crate(file_id) {
            Some(krate) => krate.to_display_target(sema.db),
            None => return,
        };
        let tokens = root.descendants_with_tokens().filter_map(|it| match it {
            syntax::NodeOrToken::Node(_) => None,
            syntax::NodeOrToken::Token(it) => Some(it),
        });
        let hover_config = HoverConfig {
            links_in_hover: true,
            memory_layout: None,
            documentation: true,
            keywords: true,
            format: crate::HoverDocFormat::Markdown,
            max_trait_assoc_items_count: None,
            max_fields_count: Some(5),
            max_enum_variants_count: Some(5),
            max_subst_ty_len: SubstTyLen::Unlimited,
            show_drop_glue: true,
        };
        let tokens = tokens.filter(|token| {
            matches!(
                token.kind(),
                IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | T![super] | T![crate] | T![Self]
            )
        });
        let mut result = StaticIndexedFile { file_id, inlay_hints, folds, tokens: vec![] };

        let mut add_token = |def: Definition, range: TextRange, scope_node: &SyntaxNode| {
            let id = if let Some(it) = self.def_map.get(&def) {
                *it
            } else {
                let it = self.tokens.insert(TokenStaticData {
                    documentation: documentation_for_definition(&sema, def, scope_node),
                    hover: Some(hover_for_definition(
                        &sema,
                        file_id,
                        def,
                        None,
                        scope_node,
                        None,
                        false,
                        &hover_config,
                        edition,
                        display_target,
                    )),
                    definition: def.try_to_nav(self.db).map(UpmappingResult::call_site).map(|it| {
                        FileRange { file_id: it.file_id, range: it.focus_or_full_range() }
                    }),
                    references: vec![],
                    moniker: current_crate.and_then(|cc| def_to_moniker(self.db, def, cc)),
                    display_name: def
                        .name(self.db)
                        .map(|name| name.display(self.db, edition).to_string()),
                    signature: Some(def.label(self.db, display_target)),
                    kind: def_to_kind(self.db, def),
                });
                self.def_map.insert(def, it);
                it
            };
            let token = self.tokens.get_mut(id).unwrap();
            token.references.push(ReferenceData {
                range: FileRange { range, file_id },
                is_definition: match def.try_to_nav(self.db).map(UpmappingResult::call_site) {
                    Some(it) => it.file_id == file_id && it.focus_or_full_range() == range,
                    None => false,
                },
            });
            result.tokens.push((range, id));
        };

        if let Some(module) = sema.file_to_module_def(file_id) {
            let def = Definition::Module(module);
            let range = root.text_range();
            add_token(def, range, &root);
        }

        for token in tokens {
            let range = token.text_range();
            let node = token.parent().unwrap();
            match get_definitions(&sema, token.clone()) {
                Some(it) => {
                    for i in it {
                        add_token(i, range, &node);
                    }
                }
                None => continue,
            };
        }
        self.files.push(result);
    }

    pub fn compute<'a>(
        analysis: &'a Analysis,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) -> StaticIndex<'a> {
        let db = &analysis.db;
        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source_file_id(db).original_file(db);
            let source_root = db.file_source_root(file_id.file_id(&analysis.db)).source_root_id(db);
            let source_root = db.source_root(source_root).source_root(db);
            let is_vendored = match vendored_libs_config {
                VendoredLibrariesConfig::Included { workspace_root } => source_root
                    .path_for_file(&file_id.file_id(&analysis.db))
                    .is_some_and(|module_path| module_path.starts_with(workspace_root)),
                VendoredLibrariesConfig::Excluded => false,
            };

            !source_root.is_library || is_vendored
        });
        let mut this = StaticIndex {
            files: vec![],
            tokens: Default::default(),
            analysis,
            db,
            def_map: Default::default(),
        };
        let mut visited_files = FxHashSet::default();
        for module in work {
            let file_id = module.definition_source_file_id(db).original_file(db);
            if visited_files.contains(&file_id) {
                continue;
            }
            this.add_file(file_id.file_id(&analysis.db));
            // mark the file
            visited_files.insert(file_id);
        }
        this
    }
}

#[cfg(test)]
mod tests {
    use crate::{StaticIndex, fixture};
    use ide_db::{FileRange, FxHashMap, FxHashSet, base_db::VfsPath};
    use syntax::TextSize;

    use super::VendoredLibrariesConfig;

    fn check_all_ranges(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashSet<_> = ranges.iter().map(|it| it.0).collect();
        for f in s.files {
            for (range, _) in f.tokens {
                if range.start() == TextSize::from(0) {
                    // ignore whole file range corresponding to module definition
                    continue;
                }
                let it = FileRange { file_id: f.file_id, range };
                if !range_set.contains(&it) {
                    panic!("additional range {it:?}");
                }
                range_set.remove(&it);
            }
        }
        if !range_set.is_empty() {
            panic!("unfound ranges {range_set:?}");
        }
    }

    #[track_caller]
    fn check_definitions(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashSet<_> = ranges.iter().map(|it| it.0).collect();
        for (_, t) in s.tokens.iter() {
            if let Some(t) = t.definition {
                if t.range.start() == TextSize::from(0) {
                    // ignore definitions that are whole of file
                    continue;
                }
                if !range_set.contains(&t) {
                    panic!("additional definition {t:?}");
                }
                range_set.remove(&t);
            }
        }
        if !range_set.is_empty() {
            panic!("unfound definitions {range_set:?}");
        }
    }

    #[track_caller]
    fn check_references(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashMap<_, i32> = ranges.iter().map(|it| (it.0, 0)).collect();

        // Make sure that all references have at least one range. We use a HashMap instead of a
        // a HashSet so that we can have more than one reference at the same range.
        for (_, t) in s.tokens.iter() {
            for r in &t.references {
                if r.is_definition {
                    continue;
                }
                if r.range.range.start() == TextSize::from(0) {
                    // ignore whole file range corresponding to module definition
                    continue;
                }
                match range_set.entry(r.range) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let count = entry.get_mut();
                        *count += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(_) => {
                        panic!("additional reference {r:?}");
                    }
                }
            }
        }
        for (range, count) in range_set.iter() {
            if *count == 0 {
                panic!("unfound reference {range:?}");
            }
        }
    }

    #[test]
    fn field_initialization() {
        check_references(
            r#"
struct Point {
    x: f64,
     //^^^
    y: f64,
     //^^^
}
    fn foo() {
        let x = 5.;
        let y = 10.;
        let mut p = Point { x, y };
                  //^^^^^   ^  ^
        p.x = 9.;
      //^ ^
        p.y = 10.;
      //^ ^
    }
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn struct_and_enum() {
        check_all_ranges(
            r#"
struct Foo;
     //^^^
enum E { X(Foo) }
   //^   ^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
        check_definitions(
            r#"
struct Foo;
     //^^^
enum E { X(Foo) }
   //^   ^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );

        check_references(
            r#"
struct Foo;
enum E { X(Foo) }
   //      ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn multi_crate() {
        check_definitions(
            r#"
//- /workspace/main.rs crate:main deps:foo


use foo::func;

fn main() {
 //^^^^
    func();
}
//- /workspace/foo/lib.rs crate:foo

pub func() {

}
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn vendored_crate() {
        check_all_ranges(
            r#"
//- /workspace/main.rs crate:main deps:external,vendored
struct Main(i32);
     //^^^^ ^^^

//- /external/lib.rs new_source_root:library crate:external@0.1.0,https://a.b/foo.git library
struct ExternalLibrary(i32);

//- /workspace/vendored/lib.rs new_source_root:library crate:vendored@0.1.0,https://a.b/bar.git library
struct VendoredLibrary(i32);
     //^^^^^^^^^^^^^^^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn vendored_crate_excluded() {
        check_all_ranges(
            r#"
//- /workspace/main.rs crate:main deps:external,vendored
struct Main(i32);
     //^^^^ ^^^

//- /external/lib.rs new_source_root:library crate:external@0.1.0,https://a.b/foo.git library
struct ExternalLibrary(i32);

//- /workspace/vendored/lib.rs new_source_root:library crate:vendored@0.1.0,https://a.b/bar.git library
struct VendoredLibrary(i32);
"#,
            VendoredLibrariesConfig::Excluded,
        )
    }

    #[test]
    fn derives() {
        check_all_ranges(
            r#"
//- minicore:derive
#[rustc_builtin_macro]
//^^^^^^^^^^^^^^^^^^^
pub macro Copy {}
        //^^^^
#[derive(Copy)]
//^^^^^^ ^^^^
struct Hello(i32);
     //^^^^^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }
}
