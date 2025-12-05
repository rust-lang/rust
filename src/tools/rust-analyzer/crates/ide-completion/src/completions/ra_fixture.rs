//! Injected completions for `#[rust_analyzer::rust_fixture]`.

use hir::FilePositionWrapper;
use ide_db::{
    impl_empty_upmap_from_ra_fixture,
    ra_fixture::{RaFixtureAnalysis, UpmapFromRaFixture},
};
use syntax::ast;

use crate::{
    CompletionItemKind, CompletionItemRefMode, CompletionRelevance, completions::Completions,
    context::CompletionContext, item::CompletionItemLabel,
};

pub(crate) fn complete_ra_fixture(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    original: &ast::String,
    expanded: &ast::String,
) -> Option<()> {
    let analysis = RaFixtureAnalysis::analyze_ra_fixture(
        &ctx.sema,
        original.clone(),
        expanded,
        ctx.config.minicore,
        &mut |_| {},
    )?;
    let (virtual_file_id, virtual_offset) = analysis.map_offset_down(ctx.position.offset)?;
    let completions = hir::attach_db_allow_change(&analysis.db, || {
        crate::completions(
            &analysis.db,
            ctx.config,
            FilePositionWrapper { file_id: virtual_file_id, offset: virtual_offset },
            ctx.trigger_character,
        )
    })?;
    let completions =
        completions.upmap_from_ra_fixture(&analysis, virtual_file_id, ctx.position.file_id).ok()?;
    acc.add_many(completions);
    Some(())
}

impl_empty_upmap_from_ra_fixture!(
    CompletionItemLabel,
    CompletionItemKind,
    CompletionRelevance,
    CompletionItemRefMode,
);

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::tests::check;

    #[test]
    fn it_works() {
        check(
            r##"
fn fixture(#[rust_analyzer::rust_fixture] ra_fixture: &str) {}

fn foo() {
    fixture(r#"
fn complete_me() {}

fn baz() {
    let foo_bar_baz = 123;
    f$0
}
    "#);
}
        "##,
            expect![[r#"
                fn baz()         fn()
                fn complete_me() fn()
                lc foo_bar_baz    i32
                bt u32            u32
                kw async
                kw const
                kw crate::
                kw enum
                kw extern
                kw false
                kw fn
                kw for
                kw if
                kw if let
                kw impl
                kw impl for
                kw let
                kw letm
                kw loop
                kw match
                kw mod
                kw return
                kw self::
                kw static
                kw struct
                kw trait
                kw true
                kw type
                kw union
                kw unsafe
                kw use
                kw while
                kw while let
                sn macro_rules
                sn pd
                sn ppd
            "#]],
        );
    }
}
