use hir::{HasSource, InFile, InRealFile, Semantics};
use ide_db::{
    FileId, FilePosition, FileRange, FxIndexSet, RootDatabase, defs::Definition,
    helpers::visit_file_defs,
};
use itertools::Itertools;
use syntax::{AstNode, TextRange, ast::HasName};

use crate::{
    NavigationTarget, RunnableKind,
    annotations::fn_references::find_all_methods,
    goto_implementation::goto_implementation,
    navigation_target,
    references::find_all_refs,
    runnables::{Runnable, runnables},
};

mod fn_references;

// Feature: Annotations
//
// Provides user with annotations above items for looking up references or impl blocks
// and running/debugging binaries.
//
// ![Annotations](https://user-images.githubusercontent.com/48062697/113020672-b7c34f00-917a-11eb-8f6e-858735660a0e.png)
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct Annotation {
    pub range: TextRange,
    pub kind: AnnotationKind,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub enum AnnotationKind {
    Runnable(Runnable),
    HasImpls { pos: FilePosition, data: Option<Vec<NavigationTarget>> },
    HasReferences { pos: FilePosition, data: Option<Vec<FileRange>> },
}

pub struct AnnotationConfig {
    pub binary_target: bool,
    pub annotate_runnables: bool,
    pub annotate_impls: bool,
    pub annotate_references: bool,
    pub annotate_method_references: bool,
    pub annotate_enum_variant_references: bool,
    pub location: AnnotationLocation,
}

pub enum AnnotationLocation {
    AboveName,
    AboveWholeItem,
}

pub(crate) fn annotations(
    db: &RootDatabase,
    config: &AnnotationConfig,
    file_id: FileId,
) -> Vec<Annotation> {
    let mut annotations = FxIndexSet::default();

    if config.annotate_runnables {
        for runnable in runnables(db, file_id) {
            if should_skip_runnable(&runnable.kind, config.binary_target) {
                continue;
            }

            let range = runnable.nav.focus_or_full_range();

            annotations.insert(Annotation { range, kind: AnnotationKind::Runnable(runnable) });
        }
    }

    let mk_ranges = |(range, focus): (_, Option<_>)| {
        let cmd_target: TextRange = focus.unwrap_or(range);
        let annotation_range = match config.location {
            AnnotationLocation::AboveName => cmd_target,
            AnnotationLocation::AboveWholeItem => range,
        };
        let target_pos = FilePosition { file_id, offset: cmd_target.start() };
        (annotation_range, target_pos)
    };

    visit_file_defs(&Semantics::new(db), file_id, &mut |def| {
        let range = match def {
            Definition::Const(konst) if config.annotate_references => {
                konst.source(db).and_then(|node| name_range(db, node, file_id))
            }
            Definition::Trait(trait_) if config.annotate_references || config.annotate_impls => {
                trait_.source(db).and_then(|node| name_range(db, node, file_id))
            }
            Definition::Adt(adt) => match adt {
                hir::Adt::Enum(enum_) => {
                    if config.annotate_enum_variant_references {
                        enum_
                            .variants(db)
                            .into_iter()
                            .filter_map(|variant| {
                                variant.source(db).and_then(|node| name_range(db, node, file_id))
                            })
                            .for_each(|range| {
                                let (annotation_range, target_position) = mk_ranges(range);
                                annotations.insert(Annotation {
                                    range: annotation_range,
                                    kind: AnnotationKind::HasReferences {
                                        pos: target_position,
                                        data: None,
                                    },
                                });
                            })
                    }
                    if config.annotate_references || config.annotate_impls {
                        enum_.source(db).and_then(|node| name_range(db, node, file_id))
                    } else {
                        None
                    }
                }
                _ => {
                    if config.annotate_references || config.annotate_impls {
                        adt.source(db).and_then(|node| name_range(db, node, file_id))
                    } else {
                        None
                    }
                }
            },
            _ => None,
        };

        let range = match range {
            Some(range) => range,
            None => return,
        };
        let (annotation_range, target_pos) = mk_ranges(range);
        if config.annotate_impls && !matches!(def, Definition::Const(_)) {
            annotations.insert(Annotation {
                range: annotation_range,
                kind: AnnotationKind::HasImpls { pos: target_pos, data: None },
            });
        }

        if config.annotate_references {
            annotations.insert(Annotation {
                range: annotation_range,
                kind: AnnotationKind::HasReferences { pos: target_pos, data: None },
            });
        }

        fn name_range<T: HasName>(
            db: &RootDatabase,
            node: InFile<T>,
            source_file_id: FileId,
        ) -> Option<(TextRange, Option<TextRange>)> {
            if let Some(name) = node.value.name().map(|name| name.syntax().text_range()) {
                // if we have a name, try mapping that out of the macro expansion as we can put the
                // annotation on that name token
                // See `test_no_annotations_macro_struct_def` vs `test_annotations_macro_struct_def_call_site`
                let res = navigation_target::orig_range_with_focus_r(
                    db,
                    node.file_id,
                    node.value.syntax().text_range(),
                    Some(name),
                );
                if res.call_site.0.file_id == source_file_id {
                    if let Some(name_range) = res.call_site.1 {
                        return Some((res.call_site.0.range, Some(name_range)));
                    }
                }
            };
            // otherwise try upmapping the entire node out of attributes
            let InRealFile { file_id, value } = node.original_ast_node_rooted(db)?;
            if file_id.file_id(db) == source_file_id {
                Some((
                    value.syntax().text_range(),
                    value.name().map(|name| name.syntax().text_range()),
                ))
            } else {
                None
            }
        }
    });

    if config.annotate_method_references {
        annotations.extend(find_all_methods(db, file_id).into_iter().map(|range| {
            let (annotation_range, target_range) = mk_ranges(range);
            Annotation {
                range: annotation_range,
                kind: AnnotationKind::HasReferences { pos: target_range, data: None },
            }
        }));
    }

    annotations
        .into_iter()
        .sorted_by_key(|a| {
            (a.range.start(), a.range.end(), matches!(a.kind, AnnotationKind::Runnable(..)))
        })
        .collect()
}

pub(crate) fn resolve_annotation(db: &RootDatabase, mut annotation: Annotation) -> Annotation {
    match annotation.kind {
        AnnotationKind::HasImpls { pos, ref mut data } => {
            *data = goto_implementation(db, pos).map(|range| range.info);
        }
        AnnotationKind::HasReferences { pos, ref mut data } => {
            *data = find_all_refs(&Semantics::new(db), pos, None).map(|result| {
                result
                    .into_iter()
                    .flat_map(|res| res.references)
                    .flat_map(|(file_id, access)| {
                        access.into_iter().map(move |(range, _)| FileRange { file_id, range })
                    })
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
    use expect_test::{Expect, expect};

    use crate::{Annotation, AnnotationConfig, fixture};

    use super::AnnotationLocation;

    const DEFAULT_CONFIG: AnnotationConfig = AnnotationConfig {
        binary_target: true,
        annotate_runnables: true,
        annotate_impls: true,
        annotate_references: true,
        annotate_method_references: true,
        annotate_enum_variant_references: true,
        location: AnnotationLocation::AboveName,
    };

    fn check_with_config(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        expect: Expect,
        config: &AnnotationConfig,
    ) {
        let (analysis, file_id) = fixture::file(ra_fixture);

        let annotations: Vec<Annotation> = analysis
            .annotations(config, file_id)
            .unwrap()
            .into_iter()
            .map(|annotation| analysis.resolve_annotation(annotation).unwrap())
            .collect();

        expect.assert_debug_eq(&annotations);
    }

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
        check_with_config(ra_fixture, expect, &DEFAULT_CONFIG);
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
                        range: 6..10,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 6,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
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
                        range: 30..36,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
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
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 53..57,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
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
                        range: 7..11,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
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
                        range: 7..11,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
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
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 17..21,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
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
                        range: 7..11,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
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
                        range: 7..11,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 57..61,
                                    },
                                    FileRangeWrapper {
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
                        range: 20..31,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
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
                        range: 20..31,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 20,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
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
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 69..73,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
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
                        range: 3..7,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 3..7,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
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
                        range: 7..11,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
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
                        range: 7..11,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 7,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 19..23,
                                    },
                                    FileRangeWrapper {
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
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 33,
                            },
                            data: Some(
                                [
                                    FileRangeWrapper {
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
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 61..65,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
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
                        range: 3..7,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
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
                    Annotation {
                        range: 3..7,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
                    },
                    Annotation {
                        range: 18..23,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
                                nav: NavigationTarget {
                                    file_id: FileId(
                                        0,
                                    ),
                                    full_range: 14..64,
                                    focus_range: 18..23,
                                    name: "tests",
                                    kind: Module,
                                    description: "mod tests",
                                },
                                kind: TestMod {
                                    path: "tests",
                                },
                                cfg: None,
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
                    },
                    Annotation {
                        range: 45..57,
                        kind: Runnable(
                            Runnable {
                                use_name_in_title: false,
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
                                update_test: UpdateTest {
                                    expect_test: false,
                                    insta: false,
                                    snapbox: false,
                                },
                            },
                        ),
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_no_annotations_outside_module_tree() {
        check(
            r#"
//- /foo.rs
struct Foo;
//- /lib.rs
// this file comes last since `check` checks the first file only
"#,
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
    fn test_no_annotations_macro_struct_def() {
        check(
            r#"
//- /lib.rs
macro_rules! m {
    () => {
        struct A {}
    };
}

m!();
"#,
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
    fn test_annotations_macro_struct_def_call_site() {
        check(
            r#"
//- /lib.rs
macro_rules! m {
    ($name:ident) => {
        struct $name {}
    };
}

m! {
    Name
};
"#,
            expect![[r#"
                [
                    Annotation {
                        range: 83..87,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 83,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                    Annotation {
                        range: 83..87,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 83,
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
    fn test_annotations_appear_above_whole_item_when_configured_to_do_so() {
        check_with_config(
            r#"
/// This is a struct named Foo, obviously.
#[derive(Clone)]
struct Foo;
"#,
            expect![[r#"
                [
                    Annotation {
                        range: 0..71,
                        kind: HasImpls {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 67,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                    Annotation {
                        range: 0..71,
                        kind: HasReferences {
                            pos: FilePositionWrapper {
                                file_id: FileId(
                                    0,
                                ),
                                offset: 67,
                            },
                            data: Some(
                                [],
                            ),
                        },
                    },
                ]
            "#]],
            &AnnotationConfig { location: AnnotationLocation::AboveWholeItem, ..DEFAULT_CONFIG },
        );
    }
}
