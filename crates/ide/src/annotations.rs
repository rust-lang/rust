use hir::Semantics;
use ide_db::{
    base_db::{FileId, FilePosition, FileRange, SourceDatabase},
    RootDatabase, SymbolKind,
};
use syntax::TextRange;

use crate::{
    file_structure::file_structure,
    fn_references::find_all_methods,
    goto_implementation::goto_implementation,
    references::find_all_refs,
    runnables::{runnables, Runnable},
    NavigationTarget, RunnableKind,
};

// Feature: Annotations
//
// Provides user with annotations above items for looking up references or impl blocks
// and running/debugging binaries.
pub struct Annotation {
    pub range: TextRange,
    pub kind: AnnotationKind,
}

pub enum AnnotationKind {
    Runnable { debug: bool, runnable: Runnable },
    HasImpls { position: FilePosition, data: Option<Vec<NavigationTarget>> },
    HasReferences { position: FilePosition, data: Option<Vec<FileRange>> },
}

pub struct AnnotationConfig {
    pub binary_target: bool,
    pub annotate_runnables: bool,
    pub annotate_impls: bool,
    pub annotate_references: bool,
    pub annotate_method_references: bool,
    pub run: bool,
    pub debug: bool,
}

pub(crate) fn annotations(
    db: &RootDatabase,
    file_id: FileId,
    config: AnnotationConfig,
) -> Vec<Annotation> {
    let mut annotations = Vec::default();

    if config.annotate_runnables {
        for runnable in runnables(db, file_id) {
            if !matches!(runnable.kind, RunnableKind::Bin) || !config.binary_target {
                continue;
            }

            let action = runnable.action();
            let range = runnable.nav.full_range;

            if config.run {
                annotations.push(Annotation {
                    range,
                    // FIXME: This one allocates without reason if run is enabled, but debug is disabled
                    kind: AnnotationKind::Runnable { debug: false, runnable: runnable.clone() },
                });
            }

            if action.debugee && config.debug {
                annotations.push(Annotation {
                    range,
                    kind: AnnotationKind::Runnable { debug: true, runnable },
                });
            }
        }
    }

    file_structure(&db.parse(file_id).tree())
        .into_iter()
        .filter(|node| {
            matches!(
                node.kind,
                SymbolKind::Trait
                    | SymbolKind::Struct
                    | SymbolKind::Enum
                    | SymbolKind::Union
                    | SymbolKind::Const
            )
        })
        .for_each(|node| {
            if config.annotate_impls && node.kind != SymbolKind::Const {
                annotations.push(Annotation {
                    range: node.node_range,
                    kind: AnnotationKind::HasImpls {
                        position: FilePosition { file_id, offset: node.navigation_range.start() },
                        data: None,
                    },
                });
            }

            if config.annotate_references {
                annotations.push(Annotation {
                    range: node.node_range,
                    kind: AnnotationKind::HasReferences {
                        position: FilePosition { file_id, offset: node.navigation_range.start() },
                        data: None,
                    },
                });
            }
        });

    if config.annotate_method_references {
        annotations.extend(find_all_methods(db, file_id).into_iter().map(|method| Annotation {
            range: method.range,
            kind: AnnotationKind::HasReferences {
                position: FilePosition { file_id, offset: method.range.start() },
                data: None,
            },
        }));
    }

    annotations
}

pub(crate) fn resolve_annotation(db: &RootDatabase, mut annotation: Annotation) -> Annotation {
    match annotation.kind {
        AnnotationKind::HasImpls { position, ref mut data } => {
            *data = goto_implementation(db, position).map(|range| range.info);
        }
        AnnotationKind::HasReferences { position, ref mut data } => {
            *data = find_all_refs(&Semantics::new(db), position, None).map(|result| {
                result
                    .references
                    .into_iter()
                    .map(|(_, access)| access.into_iter())
                    .flatten()
                    .map(|(range, _)| FileRange { file_id: position.file_id, range })
                    .collect()
            });
        }
        _ => {}
    };

    annotation
}
