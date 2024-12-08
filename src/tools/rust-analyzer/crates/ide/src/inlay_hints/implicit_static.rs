//! Implementation of "implicit static" inlay hints:
//! ```no_run
//! static S: &/* 'static */str = "";
//! ```
use either::Either;
use ide_db::famous_defs::FamousDefs;
use ide_db::text_edit::TextEdit;
use span::EditionedFileId;
use syntax::{
    ast::{self, AstNode},
    SyntaxKind,
};

use crate::{InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind, LifetimeElisionHints};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(_sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    _file_id: EditionedFileId,
    statik_or_const: Either<ast::Static, ast::Const>,
) -> Option<()> {
    if config.lifetime_elision_hints != LifetimeElisionHints::Always {
        return None;
    }

    if let Either::Right(it) = &statik_or_const {
        if ast::AssocItemList::can_cast(
            it.syntax().parent().map_or(SyntaxKind::EOF, |it| it.kind()),
        ) {
            return None;
        }
    }

    if let Some(ast::Type::RefType(ty)) = statik_or_const.either(|it| it.ty(), |it| it.ty()) {
        if ty.lifetime().is_none() {
            let t = ty.amp_token()?;
            acc.push(InlayHint {
                range: t.text_range(),
                kind: InlayKind::Lifetime,
                label: "'static".into(),
                text_edit: Some(TextEdit::insert(t.text_range().start(), "'static ".into())),
                position: InlayHintPosition::After,
                pad_left: false,
                pad_right: true,
                resolve_parent: None,
            });
        }
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        inlay_hints::tests::{check_with_config, TEST_CONFIG},
        InlayHintsConfig, LifetimeElisionHints,
    };

    #[test]
    fn hints_lifetimes_static() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
trait Trait {}
static S: &str = "";
//        ^'static
const C: &str = "";
//       ^'static
const C: &dyn Trait = panic!();
//       ^'static

impl () {
    const C: &str = "";
    const C: &dyn Trait = panic!();
}
"#,
        );
    }
}
