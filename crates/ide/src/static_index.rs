//! This module provides `StaticIndex` which is used for powering
//! read-only code browsers and emitting LSIF

use std::collections::HashMap;

use hir::{db::HirDatabase, Crate, Module};
use ide_db::base_db::{FileId, SourceDatabaseExt};
use ide_db::RootDatabase;
use ide_db::defs::Definition;
use rustc_hash::FxHashSet;
use syntax::TextRange;
use syntax::{AstNode, SyntaxKind::*, T};

use crate::hover::{get_definition_of_token, hover_for_definition};
use crate::{Analysis, Cancellable, Fold, HoverConfig, HoverDocFormat, HoverResult};

/// A static representation of fully analyzed source code.
///
/// The intended use-case is powering read-only code browsers and emitting LSIF
pub struct StaticIndex<'a> {
    pub files: Vec<StaticIndexedFile>,
    pub tokens: TokenStore,
    analysis: &'a Analysis,
    db: &'a RootDatabase,
    def_map: HashMap<Definition, TokenId>,
}

pub struct TokenStaticData {
    pub hover: Option<HoverResult>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(usize);

#[derive(Default)]
pub struct TokenStore(Vec<TokenStaticData>);

impl TokenStore {
    pub fn insert(&mut self, data: TokenStaticData) -> TokenId {
        let id = TokenId(self.0.len());
        self.0.push(data);
        id
    }

    pub fn get(&self, id: TokenId) -> Option<&TokenStaticData> {
        self.0.get(id.0)
    }
    
    pub fn iter(self) -> impl Iterator<Item=(TokenId, TokenStaticData)> {
        self.0.into_iter().enumerate().map(|(i, x)| {
            (TokenId(i), x)
        })
    }
}

pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,
    pub tokens: Vec<(TextRange, TokenId)>,
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

impl StaticIndex<'_> {
    fn add_file(&mut self, file_id: FileId) -> Cancellable<()> {
        let folds = self.analysis.folding_ranges(file_id)?;
        // hovers
        let sema = hir::Semantics::new(self.db);
        let tokens_or_nodes = sema.parse(file_id).syntax().clone();
        let tokens = tokens_or_nodes.descendants_with_tokens().filter_map(|x| match x {
            syntax::NodeOrToken::Node(_) => None,
            syntax::NodeOrToken::Token(x) => Some(x),
        });
        let hover_config =
            HoverConfig { links_in_hover: true, documentation: Some(HoverDocFormat::Markdown) };
        let tokens = tokens
            .filter(|token| match token.kind() {
                IDENT
                | INT_NUMBER
                | LIFETIME_IDENT
                | T![self]
                | T![super]
                | T![crate] => true,
                _ => false,
            });
        let mut result = StaticIndexedFile {
            file_id,
            folds,
            tokens: vec![],
        };
        for token in tokens {
            let range = token.text_range();
            let node = token.parent().unwrap();
            let def = get_definition_of_token(self.db, &sema, &sema.descend_into_macros(token), file_id, range.start(), &mut None);
            let def = if let Some(x) = def {
                x
            } else {
                continue;
            };
            let id = if let Some(x) = self.def_map.get(&def) {
                *x
            } else {
                let x = self.tokens.insert(TokenStaticData {
                    hover: hover_for_definition(self.db, file_id, &sema, def, node, &hover_config),
                });
                self.def_map.insert(def, x);
                x
            };
            result.tokens.push((range, id));
        }
        self.files.push(result);
        Ok(())
    }
    
    pub fn compute<'a>(db: &'a RootDatabase, analysis: &'a Analysis) -> Cancellable<StaticIndex<'a>> {
        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source(db).file_id.original_file(db);
            let source_root = db.file_source_root(file_id);
            let source_root = db.source_root(source_root);
            !source_root.is_library
        });
        let mut this = StaticIndex {
            files: vec![],
            tokens: Default::default(),
            analysis, db,
            def_map: Default::default(),
        };
        let mut visited_files = FxHashSet::default();
        for module in work {
            let file_id = module.definition_source(db).file_id.original_file(db);
            if visited_files.contains(&file_id) {
                continue;
            }
            this.add_file(file_id)?;
            // mark the file
            visited_files.insert(file_id);
        }
        //eprintln!("{:#?}", token_map);
        Ok(this)
    }
}
