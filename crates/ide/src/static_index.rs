//! This module provides `StaticIndex` which is used for powering
//! read-only code browsers and emitting LSIF

use hir::{db::HirDatabase, Crate, Module};
use ide_db::base_db::{FileId, FileRange, SourceDatabaseExt};
use ide_db::RootDatabase;
use rustc_hash::FxHashSet;
use syntax::TextRange;
use syntax::{AstNode, SyntaxKind::*, T};

use crate::{Analysis, Cancellable, Fold, HoverConfig, HoverDocFormat, HoverResult};

/// A static representation of fully analyzed source code.
///
/// The intended use-case is powering read-only code browsers and emitting LSIF
pub struct StaticIndex {
    pub files: Vec<StaticIndexedFile>,
}

pub struct TokenStaticData {
    pub range: TextRange,
    pub hover: Option<HoverResult>,
}

pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,
    pub tokens: Vec<TokenStaticData>,
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

impl StaticIndex {
    pub fn compute(db: &RootDatabase, analysis: &Analysis) -> Cancellable<StaticIndex> {
        let work = all_modules(db).into_iter().filter(|module| {
            let file_id = module.definition_source(db).file_id.original_file(db);
            let source_root = db.file_source_root(file_id);
            let source_root = db.source_root(source_root);
            !source_root.is_library
        });

        let mut visited_files = FxHashSet::default();
        let mut result_files = Vec::<StaticIndexedFile>::new();
        for module in work {
            let file_id = module.definition_source(db).file_id.original_file(db);
            if visited_files.contains(&file_id) {
                continue;
            }
            let folds = analysis.folding_ranges(file_id)?;
            // hovers
            let sema = hir::Semantics::new(db);
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
                    | T![crate]
                    | T!['(']
                    | T![')'] => true,
                    _ => false,
                })
                .map(|token| {
                    let range = token.text_range();
                    let hover = analysis
                        .hover(
                            &hover_config,
                            FileRange {
                                file_id,
                                range: TextRange::new(range.start(), range.start()),
                            },
                        )?
                        .map(|x| x.info);
                    Ok(TokenStaticData { range, hover })
                })
                .collect::<Result<Vec<_>, _>>()?;
            result_files.push(StaticIndexedFile { file_id, folds, tokens });
            // mark the file
            visited_files.insert(file_id);
        }
        Ok(StaticIndex { files: result_files })
    }
}
