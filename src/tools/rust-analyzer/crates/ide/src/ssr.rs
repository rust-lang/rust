//! This module provides an SSR assist. It is not desirable to include this
//! assist in ide_assists because that would require the ide_assists crate
//! depend on the ide_ssr crate.

use ide_assists::{Assist, AssistId, AssistResolveStrategy, GroupLabel};
use ide_db::{FileRange, RootDatabase, label::Label, source_change::SourceChange};

pub(crate) fn ssr_assists(
    db: &RootDatabase,
    resolve: &AssistResolveStrategy,
    frange: FileRange,
) -> Vec<Assist> {
    let mut ssr_assists = Vec::with_capacity(2);

    let (match_finder, comment_range) = match ide_ssr::ssr_from_comment(db, frange) {
        Some(ssr_data) => ssr_data,
        None => return ssr_assists,
    };
    let id = AssistId::refactor_rewrite("ssr");

    let (source_change_for_file, source_change_for_workspace) = if resolve.should_resolve(&id) {
        let edits = match_finder.edits();

        let source_change_for_file = {
            let text_edit_for_file = edits.get(&frange.file_id).cloned().unwrap_or_default();
            SourceChange::from_text_edit(frange.file_id, text_edit_for_file)
        };

        let source_change_for_workspace = SourceChange::from_iter(match_finder.edits());

        (Some(source_change_for_file), Some(source_change_for_workspace))
    } else {
        (None, None)
    };

    let assists = vec![
        ("Apply SSR in file", source_change_for_file),
        ("Apply SSR in workspace", source_change_for_workspace),
    ];

    for (label, source_change) in assists.into_iter() {
        let assist = Assist {
            id,
            label: Label::new(label.to_owned()),
            group: Some(GroupLabel("Apply SSR".into())),
            target: comment_range,
            source_change,
            command: None,
        };

        ssr_assists.push(assist);
    }

    ssr_assists
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use ide_assists::{Assist, AssistResolveStrategy};
    use ide_db::{
        FileRange, FxHashSet, RootDatabase, base_db::salsa::Durability,
        symbol_index::SymbolsDatabase,
    };
    use test_fixture::WithFixture;
    use triomphe::Arc;

    use super::ssr_assists;

    fn get_assists(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        resolve: AssistResolveStrategy,
    ) -> Vec<Assist> {
        let (mut db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(ra_fixture);
        let mut local_roots = FxHashSet::default();
        local_roots.insert(test_fixture::WORKSPACE);
        db.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);
        ssr_assists(
            &db,
            &resolve,
            FileRange { file_id: file_id.file_id(&db), range: range_or_offset.into() },
        )
    }

    #[test]
    fn not_applicable_comment_not_ssr() {
        let ra_fixture = r#"
            //- /lib.rs

            // This is foo $0
            fn foo() {}
            "#;
        let assists = get_assists(ra_fixture, AssistResolveStrategy::All);

        assert_eq!(0, assists.len());
    }

    #[test]
    fn resolve_edits_true() {
        let assists = get_assists(
            r#"
            //- /lib.rs
            mod bar;

            // 2 ==>> 3$0
            fn foo() { 2 }

            //- /bar.rs
            fn bar() { 2 }
            "#,
            AssistResolveStrategy::All,
        );

        assert_eq!(2, assists.len());
        let mut assists = assists.into_iter();

        let apply_in_file_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                    None,
                ),
                label: "Apply SSR in file",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: Some(
                    SourceChange {
                        source_file_edits: {
                            FileId(
                                0,
                            ): (
                                TextEdit {
                                    indels: [
                                        Indel {
                                            insert: "3",
                                            delete: 33..34,
                                        },
                                    ],
                                    annotation: None,
                                },
                                None,
                            ),
                        },
                        file_system_edits: [],
                        is_snippet: false,
                        annotations: {},
                        next_annotation_id: 0,
                    },
                ),
                command: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_file_assist);

        let apply_in_workspace_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                    None,
                ),
                label: "Apply SSR in workspace",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: Some(
                    SourceChange {
                        source_file_edits: {
                            FileId(
                                0,
                            ): (
                                TextEdit {
                                    indels: [
                                        Indel {
                                            insert: "3",
                                            delete: 33..34,
                                        },
                                    ],
                                    annotation: None,
                                },
                                None,
                            ),
                            FileId(
                                1,
                            ): (
                                TextEdit {
                                    indels: [
                                        Indel {
                                            insert: "3",
                                            delete: 11..12,
                                        },
                                    ],
                                    annotation: None,
                                },
                                None,
                            ),
                        },
                        file_system_edits: [],
                        is_snippet: false,
                        annotations: {},
                        next_annotation_id: 0,
                    },
                ),
                command: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_workspace_assist);
    }

    #[test]
    fn resolve_edits_false() {
        let assists = get_assists(
            r#"
            //- /lib.rs
            mod bar;

            // 2 ==>> 3$0
            fn foo() { 2 }

            //- /bar.rs
            fn bar() { 2 }
            "#,
            AssistResolveStrategy::None,
        );

        assert_eq!(2, assists.len());
        let mut assists = assists.into_iter();

        let apply_in_file_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                    None,
                ),
                label: "Apply SSR in file",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: None,
                command: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_file_assist);

        let apply_in_workspace_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                    None,
                ),
                label: "Apply SSR in workspace",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: None,
                command: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_workspace_assist);
    }
}
