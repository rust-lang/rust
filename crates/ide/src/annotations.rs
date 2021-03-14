use hir::Semantics;
use ide_db::{
    base_db::{FileId, FilePosition, FileRange, SourceDatabase},
    RootDatabase, StructureNodeKind, SymbolKind,
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
                StructureNodeKind::SymbolKind(SymbolKind::Trait)
                    | StructureNodeKind::SymbolKind(SymbolKind::Struct)
                    | StructureNodeKind::SymbolKind(SymbolKind::Enum)
                    | StructureNodeKind::SymbolKind(SymbolKind::Union)
                    | StructureNodeKind::SymbolKind(SymbolKind::Const)
            )
        })
        .for_each(|node| {
            if config.annotate_impls
                && node.kind != StructureNodeKind::SymbolKind(SymbolKind::Const)
            {
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
    use expect_test::{expect, Expect};

    use crate::{fixture, Annotation, AnnotationConfig};

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);

        let annotations: Vec<Annotation> = analysis
            .annotations(
                file_id,
                AnnotationConfig {
                    binary_target: true,
                    annotate_runnables: true,
                    annotate_impls: true,
                    annotate_references: true,
                    annotate_method_references: true,
                    run: true,
                    debug: true,
                },
            )
            .unwrap()
            .into_iter()
            .map(|annotation| analysis.resolve_annotation(annotation).unwrap())
            .collect();

        expect.assert_debug_eq(&annotations);
    }

    #[test]
    fn const_annotations() {
        check(
            r#"
const DEMO: i32 = 123;

const UNUSED: i32 = 123;

fn main() {
    let hello = DEMO;
}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 50..85,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 50..85,
                                    focus_range: 53..57,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 50..85,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 50..85,
                                    focus_range: 53..57,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..22,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 6,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 78..82,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 24..48,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 30,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                    Annotation {
                        range: 53..57,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 53,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn struct_references_annotations() {
        check(
            r#"
struct Test;

fn main() {
    let test = Test;
}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 14..48,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 14..48,
                                    focus_range: 17..21,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 14..48,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 14..48,
                                    focus_range: 17..21,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasImpls {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 41..45,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 17..21,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 17,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn struct_and_trait_impls_annotations() {
        check(
            r#"
struct Test;

trait MyCoolTrait {}

impl MyCoolTrait for Test {}

fn main() {
    let test = Test;
}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 66..100,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 66..100,
                                    focus_range: 69..73,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 66..100,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 66..100,
                                    focus_range: 69..73,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasImpls {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    NavigationTarget {
                                        file_id: FileId(
                                            0,
                                        ),
                                        full_range: 36..64,
                                        focus_range: 57..61,
                                        name: "impl",
                                        kind: Impl,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 57..61,
                                    },
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 93..97,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 14..34,
                        kind: HasImpls {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 20,
                            },
                            data: Some(
                                [
                                    NavigationTarget {
                                        file_id: FileId(
                                            0,
                                        ),
                                        full_range: 36..64,
                                        focus_range: 57..61,
                                        name: "impl",
                                        kind: Impl,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 14..34,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 20,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 41..52,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 69..73,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 69,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn runnable_annotation() {
        check(
            r#"
fn main() {}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 0..12,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 3..7,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 3..7,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 3..7,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 3,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn method_annotations() {
        check(
            r#"
struct Test;

impl Test {
    fn self_by_ref(&self) {}
}

fn main() {
    Test.self_by_ref();
}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 58..95,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 58..95,
                                    focus_range: 61..65,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 58..95,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 58..95,
                                    focus_range: 61..65,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasImpls {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    NavigationTarget {
                                        file_id: FileId(
                                            0,
                                        ),
                                        full_range: 14..56,
                                        focus_range: 19..23,
                                        name: "impl",
                                        kind: Impl,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 19..23,
                                    },
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 74..78,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 33..44,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 33,
                            },
                            data: Some(
                                [
                                    FileRange {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 79..90,
                                    },
                                ],
                            ),
                        },
                    },
                    Annotation {
                        range: 61..65,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 61,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_annotations() {
        check(
            r#"
fn main() {}

mod tests {
    #[test]
    fn my_cool_test() {}
}
            "#,
            expect![[r#"
                [
                    Annotation {
                        range: 0..12,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 3..7,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 0..12,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 0..12,
                                    focus_range: 3..7,
                                    name: "main",
                                    kind: Function,
                                },
                                kind: Bin,
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 14..64,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 14..64,
                                    focus_range: 18..23,
                                    name: "tests",
                                    kind: Module,
                                },
                                kind: TestMod {
                                    path: "tests",
                                },
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 14..64,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 14..64,
                                    focus_range: 18..23,
                                    name: "tests",
                                    kind: Module,
                                },
                                kind: TestMod {
                                    path: "tests",
                                },
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 30..62,
                        kind: Runnable {
                            debug: false,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 30..62,
                                    focus_range: 45..57,
                                    name: "my_cool_test",
                                    kind: Function,
                                },
                                kind: Test {
                                    test_id: Path(
                                        "tests::my_cool_test",
                                    ),
                                    attr: TestAttr {
                                        ignore: false,
                                    },
                                },
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 30..62,
                        kind: Runnable {
                            debug: true,
                            runnable: Runnable {
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 30..62,
                                    focus_range: 45..57,
                                    name: "my_cool_test",
                                    kind: Function,
                                },
                                kind: Test {
                                    test_id: Path(
                                        "tests::my_cool_test",
                                    ),
                                    attr: TestAttr {
                                        ignore: false,
                                    },
                                },
                                cfg: None,
                            },
                        },
                    },
                    Annotation {
                        range: 3..7,
                        kind: HasReferences {
                            position: FilePosition {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 3,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
        );
    }
}
