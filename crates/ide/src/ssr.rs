//! This module provides an SSR assist. It is not desirable to include this
//! assist in ide_assists because that would require the ide_assists crate
//! depend on the ide_ssr crate.

use ide_assists::{Assist, AssistId, AssistKind, GroupLabel};
use ide_db::{base_db::FileRange, label::Label, source_change::SourceChange, RootDatabase};

pub(crate) fn add_ssr_assist(
    db: &RootDatabase,
    base: &mut Vec<Assist>,
    resolve: bool,
    frange: FileRange,
) -> Option<()> {
    let (match_finder, comment_range) = ide_ssr::ssr_from_comment(db, frange)?;

    let (source_change_for_file, source_change_for_workspace) = if resolve {
        let edits = match_finder.edits();

        let source_change_for_file = {
            let text_edit_for_file = edits.get(&frange.file_id).cloned().unwrap_or_default();
            SourceChange::from_text_edit(frange.file_id, text_edit_for_file)
        };

        let source_change_for_workspace = SourceChange::from(match_finder.edits());

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
            id: AssistId("ssr", AssistKind::RefactorRewrite),
            label: Label::new(label),
            group: Some(GroupLabel("Apply SSR".into())),
            target: comment_range,
            source_change,
        };

        base.push(assist);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use expect_test::expect;
    use ide_assists::Assist;
    use ide_db::{
        base_db::{fixture::WithFixture, salsa::Durability, FileRange},
        symbol_index::SymbolsDatabase,
        RootDatabase,
    };
    use rustc_hash::FxHashSet;

    use super::add_ssr_assist;

    fn get_assists(ra_fixture: &str, resolve: bool) -> Vec<Assist> {
        let (mut db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(ra_fixture);
        let mut local_roots = FxHashSet::default();
        local_roots.insert(ide_db::base_db::fixture::WORKSPACE);
        db.set_local_roots_with_durability(Arc::new(local_roots), Durability::HIGH);

        let mut assists = vec![];

        add_ssr_assist(
            &db,
            &mut assists,
            resolve,
            FileRange { file_id, range: range_or_offset.into() },
        );

        assists
    }

    #[test]
    fn not_applicable_comment_not_ssr() {
        let ra_fixture = r#"
            //- /lib.rs

            // This is foo $0
            fn foo() {}
            "#;
        let resolve = true;

        let assists = get_assists(ra_fixture, resolve);

        assert_eq!(0, assists.len());
    }

    #[test]
    fn resolve_edits_true() {
        let resolve = true;
        let assists = get_assists(
            r#"
            //- /lib.rs
            mod bar;

            // 2 ==>> 3$0
            fn foo() { 2 }

            //- /bar.rs
            fn bar() { 2 }
            "#,
            resolve,
        );

        assert_eq!(2, assists.len());
        let mut assists = assists.into_iter();

        let apply_in_file_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
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
                            ): TextEdit {
                                indels: [
                                    Indel {
                                        insert: "3",
                                        delete: 33..34,
                                    },
                                ],
                            },
                        },
                        file_system_edits: [],
                        is_snippet: false,
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&apply_in_file_assist);

        let apply_in_workspace_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
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
                            ): TextEdit {
                                indels: [
                                    Indel {
                                        insert: "3",
                                        delete: 33..34,
                                    },
                                ],
                            },
                            FileId(
                                1,
                            ): TextEdit {
                                indels: [
                                    Indel {
                                        insert: "3",
                                        delete: 11..12,
                                    },
                                ],
                            },
                        },
                        file_system_edits: [],
                        is_snippet: false,
                    },
                ),
            }
        "#]]
        .assert_debug_eq(&apply_in_workspace_assist);
    }

    #[test]
    fn resolve_edits_false() {
        let resolve = false;
        let assists = get_assists(
            r#"
            //- /lib.rs
            mod bar;

            // 2 ==>> 3$0
            fn foo() { 2 }

            //- /bar.rs
            fn bar() { 2 }
            "#,
            resolve,
        );

        assert_eq!(2, assists.len());
        let mut assists = assists.into_iter();

        let apply_in_file_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                ),
                label: "Apply SSR in file",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_file_assist);

        let apply_in_workspace_assist = assists.next().unwrap();
        expect![[r#"
            Assist {
                id: AssistId(
                    "ssr",
                    RefactorRewrite,
                ),
                label: "Apply SSR in workspace",
                group: Some(
                    GroupLabel(
                        "Apply SSR",
                    ),
                ),
                target: 10..21,
                source_change: None,
            }
        "#]]
        .assert_debug_eq(&apply_in_workspace_assist);
    }
}
