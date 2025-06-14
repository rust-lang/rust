//! Implementation of trait bound hints.
//!
//! Currently this renders the implied `Sized` bound.
use either::Either;
use ide_db::{famous_defs::FamousDefs, text_edit::TextEdit};

use syntax::ast::{self, AstNode};

use crate::{InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    path: Either<ast::PathType, ast::DynTraitType>,
) -> Option<()> {
    let parent = path.syntax().parent()?;
    let range = match path {
        Either::Left(path) => {
            let paren =
                parent.ancestors().take_while(|it| ast::ParenType::can_cast(it.kind())).last();
            let parent = paren.as_ref().and_then(|it| it.parent()).unwrap_or(parent);
            if ast::TypeBound::can_cast(parent.kind())
                || ast::TypeAnchor::can_cast(parent.kind())
                || ast::Impl::cast(parent)
                    .and_then(|it| it.trait_())
                    .is_some_and(|it| it.syntax() == path.syntax())
            {
                return None;
            }
            sema.resolve_trait(&path.path()?)?;
            paren.map_or_else(|| path.syntax().text_range(), |it| it.text_range())
        }
        Either::Right(dyn_) => {
            if dyn_.dyn_token().is_some() {
                return None;
            }

            dyn_.syntax().text_range()
        }
    };

    acc.push(InlayHint {
        range,
        kind: InlayKind::Dyn,
        label: InlayHintLabel::simple("dyn", None, None),
        text_edit: Some(
            config.lazy_text_edit(|| TextEdit::insert(range.start(), "dyn ".to_owned())),
        ),
        position: InlayHintPosition::Before,
        pad_left: false,
        pad_right: true,
        resolve_parent: Some(range),
    });

    Some(())
}

#[cfg(test)]
mod tests {

    use expect_test::expect;

    use crate::inlay_hints::InlayHintsConfig;

    use crate::inlay_hints::tests::{DISABLED_CONFIG, check_edit, check_with_config};

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(InlayHintsConfig { sized_bound: true, ..DISABLED_CONFIG }, ra_fixture);
    }

    #[test]
    fn path_works() {
        check(
            r#"
struct S {}
trait T {}
fn foo(_: T,  _: dyn T, _: S) {}
       // ^ dyn
fn foo(_: &T,  _: for<'a> T) {}
        // ^ dyn
                       // ^ dyn
impl T {}
  // ^ dyn
impl T for (T) {}
        // ^^^ dyn
"#,
        );
    }

    #[test]
    fn missing_dyn_bounds() {
        check(
            r#"
trait T {}
fn foo(
    _: T + T,
    // ^^^^^ dyn
    _: T + 'a,
    // ^^^^^^ dyn
    _: 'a + T,
    // ^^^^^^ dyn
    _: &(T + T)
    //   ^^^^^ dyn
    _: &mut (T + T)
    //       ^^^^^ dyn
    _: *mut (T),
    //      ^^^ dyn
) {}
"#,
        );
    }

    #[test]
    fn edit() {
        check_edit(
            DISABLED_CONFIG,
            r#"
trait T {}
fn foo(
    _: &mut T
) {}
"#,
            expect![[r#"
                trait T {}
                fn foo(
                    _: &mut dyn T
                ) {}
            "#]],
        );
    }
}
