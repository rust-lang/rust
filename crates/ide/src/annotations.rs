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
#[derive(Debug)]
pub struct Annotation {
    pub range: TextRange,
    pub kind: AnnotationKind,
}

#[derive(Debug)]
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
            if should_skip_runnable(&runnable.kind, config.binary_target) {
                continue;
            }

            let action = runnable.action();
            let range = runnable.nav.full_range;

            if action.debugee && config.debug {
                annotations.push(Annotation {
                    range,

                    // FIXME: This one allocates without reason if run is enabled, but debug is disabled
                    kind: AnnotationKind::Runnable { debug: true, runnable: runnable.clone() },
                });
            }

            if config.run {
                annotations.push(Annotation {
                    range,
                    kind: AnnotationKind::Runnable { debug: false, runnable },
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
                    .map(|(file_id, access)| {
                        access.into_iter().map(move |(range, _)| FileRange { file_id, range })
                    })
                    .flatten()
                    .collect()
            });
        }
        _ => {}
    };

    annotation
}

fn should_skip_runnable(kind: &RunnableKind, binary_target: bool) -> bool {
    match kind {
        RunnableKind::Bin => !binary_target,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::{FileId, FileRange};
    use syntax::{TextRange, TextSize};

    use crate::{fixture, Annotation, AnnotationConfig, AnnotationKind, RunnableKind};

    fn get_annotations(
        ra_fixture: &str,
        annotation_config: AnnotationConfig,
    ) -> (FileId, Vec<Annotation>) {
        let (analysis, file_id) = fixture::file(ra_fixture);

        let annotations: Vec<Annotation> = analysis
            .annotations(file_id, annotation_config)
            .unwrap()
            .into_iter()
            .map(move |annotation| analysis.resolve_annotation(annotation).unwrap())
            .collect();

        if annotations.len() == 0 {
            panic!("unresolved annotations")
        }

        (file_id, annotations)
    }

    macro_rules! check_annotation {
        ( $ra_fixture:expr, $config:expr, $item_positions:expr, $pattern:pat, $checker:expr ) => {
            let (file_id, annotations) = get_annotations($ra_fixture, $config);

            annotations.into_iter().for_each(|annotation| {
                assert!($item_positions.contains(&annotation.range));

                match annotation.kind {
                    $pattern => $checker(file_id),
                    _ => panic!("Unexpected annotation kind"),
                }
            });
        };
    }

    #[test]
    fn const_annotations() {
        check_annotation!(
            r#"
const DEMO: i32 = 123;

fn main() {
    let hello = DEMO;
}
            "#,
            AnnotationConfig {
                binary_target: false,
                annotate_runnables: false,
                annotate_impls: false,
                annotate_references: true,
                annotate_method_references: false,
                run: false,
                debug: false,
            },
            &[TextRange::new(TextSize::from(0), TextSize::from(22))],
            AnnotationKind::HasReferences { data: Some(ranges), .. },
            |file_id| assert_eq!(
                *ranges.first().unwrap(),
                FileRange {
                    file_id,
                    range: TextRange::new(TextSize::from(52), TextSize::from(56))
                }
            )
        );
    }

    #[test]
    fn unused_const_annotations() {
        check_annotation!(
            r#"
const DEMO: i32 = 123;

fn main() {}
            "#,
            AnnotationConfig {
                binary_target: false,
                annotate_runnables: false,
                annotate_impls: false,
                annotate_references: true,
                annotate_method_references: false,
                run: false,
                debug: false,
            },
            &[TextRange::new(TextSize::from(0), TextSize::from(22))],
            AnnotationKind::HasReferences { data: Some(ranges), .. },
            |_| assert_eq!(ranges.len(), 0)
        );
    }

    #[test]
    fn struct_references_annotations() {
        check_annotation!(
            r#"
struct Test;

fn main() {
    let test = Test;
}
            "#,
            AnnotationConfig {
                binary_target: false,
                annotate_runnables: false,
                annotate_impls: false,
                annotate_references: true,
                annotate_method_references: false,
                run: false,
                debug: false,
            },
            &[TextRange::new(TextSize::from(0), TextSize::from(12))],
            AnnotationKind::HasReferences { data: Some(ranges), .. },
            |file_id| assert_eq!(
                *ranges.first().unwrap(),
                FileRange {
                    file_id,
                    range: TextRange::new(TextSize::from(41), TextSize::from(45))
                }
            )
        );
    }

    #[test]
    fn struct_and_trait_impls_annotations() {
        check_annotation!(
            r#"
struct Test;

trait MyCoolTrait {}

impl MyCoolTrait for Test {}

fn main() {
    let test = Test;
}
            "#,
            AnnotationConfig {
                binary_target: false,
                annotate_runnables: false,
                annotate_impls: true,
                annotate_references: false,
                annotate_method_references: false,
                run: false,
                debug: false,
            },
            &[
                TextRange::new(TextSize::from(0), TextSize::from(12)),
                TextRange::new(TextSize::from(14), TextSize::from(34))
            ],
            AnnotationKind::HasImpls { data: Some(ranges), .. },
            |_| assert_eq!(
                ranges.first().unwrap().full_range,
                TextRange::new(TextSize::from(36), TextSize::from(64))
            )
        );
    }

    #[test]
    fn run_annotation() {
        check_annotation!(
            r#"
fn main() {}
            "#,
            AnnotationConfig {
                binary_target: true,
                annotate_runnables: true,
                annotate_impls: false,
                annotate_references: false,
                annotate_method_references: false,
                run: true,
                debug: false,
            },
            &[TextRange::new(TextSize::from(0), TextSize::from(12))],
            AnnotationKind::Runnable { debug: false, runnable },
            |_| {
                assert!(matches!(runnable.kind, RunnableKind::Bin));
                assert!(runnable.action().run_title.contains("Run"));
            }
        );
    }

    #[test]
    fn debug_annotation() {
        check_annotation!(
            r#"
fn main() {}
            "#,
            AnnotationConfig {
                binary_target: true,
                annotate_runnables: true,
                annotate_impls: false,
                annotate_references: false,
                annotate_method_references: false,
                run: false,
                debug: true,
            },
            &[TextRange::new(TextSize::from(0), TextSize::from(12))],
            AnnotationKind::Runnable { debug: true, runnable },
            |_| {
                assert!(matches!(runnable.kind, RunnableKind::Bin));
                assert!(runnable.action().debugee);
            }
        );
    }

    #[test]
    fn method_annotations() {
        // We actually want to skip `fn main` annotation, as it has no references in it
        // but just ignoring empty reference slices would lead to false-positive if something
        // goes wrong in annotation resolving mechanism. By tracking if we iterated before finding
        // an empty slice we can track if everything is settled.
        let mut iterated_once = false;

        check_annotation!(
            r#"
struct Test;

impl Test {
    fn self_by_ref(&self) {}
}

fn main() {
    Test.self_by_ref();
}
            "#,
            AnnotationConfig {
                binary_target: false,
                annotate_runnables: false,
                annotate_impls: false,
                annotate_references: false,
                annotate_method_references: true,
                run: false,
                debug: false,
            },
            &[
                TextRange::new(TextSize::from(33), TextSize::from(44)),
                TextRange::new(TextSize::from(61), TextSize::from(65))
            ],
            AnnotationKind::HasReferences { data: Some(ranges), .. },
            |file_id| {
                match ranges.as_slice() {
                    [first, ..] => {
                        assert_eq!(
                            *first,
                            FileRange {
                                file_id,
                                range: TextRange::new(TextSize::from(79), TextSize::from(90))
                            }
                        );

                        iterated_once = true;
                    }
                    [] if iterated_once => {}
                    [] => panic!("One reference was expected but not found"),
                }
            }
        );
    }
}
