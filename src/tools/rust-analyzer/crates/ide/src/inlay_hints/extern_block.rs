//! Extern block hints
use ide_db::{famous_defs::FamousDefs, text_edit::TextEdit};
use syntax::{AstNode, SyntaxToken, ast};

use crate::{InlayHint, InlayHintsConfig};

pub(super) fn extern_block_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    extern_block: ast::ExternBlock,
) -> Option<()> {
    if extern_block.unsafe_token().is_some() {
        return None;
    }
    let abi = extern_block.abi()?;
    sema.to_def(&extern_block)?;
    acc.push(InlayHint {
        range: abi.syntax().text_range(),
        position: crate::InlayHintPosition::Before,
        pad_left: false,
        pad_right: true,
        kind: crate::InlayKind::ExternUnsafety,
        label: crate::InlayHintLabel::from("unsafe"),
        text_edit: Some(config.lazy_text_edit(|| {
            TextEdit::insert(abi.syntax().text_range().start(), "unsafe ".to_owned())
        })),
        resolve_parent: Some(extern_block.syntax().text_range()),
    });
    Some(())
}

pub(super) fn fn_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    fn_: &ast::Fn,
    extern_block: &ast::ExternBlock,
) -> Option<()> {
    let implicit_unsafe = fn_.safe_token().is_none() && fn_.unsafe_token().is_none();
    if !implicit_unsafe {
        return None;
    }
    let fn_token = fn_.fn_token()?;
    if sema.to_def(fn_).is_some_and(|def| def.extern_block(sema.db).is_some()) {
        acc.push(item_hint(config, extern_block, fn_token));
    }
    Some(())
}

pub(super) fn static_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(sema, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    static_: &ast::Static,
    extern_block: &ast::ExternBlock,
) -> Option<()> {
    let implicit_unsafe = static_.safe_token().is_none() && static_.unsafe_token().is_none();
    if !implicit_unsafe {
        return None;
    }
    let static_token = static_.static_token()?;
    if sema.to_def(static_).is_some_and(|def| def.extern_block(sema.db).is_some()) {
        acc.push(item_hint(config, extern_block, static_token));
    }
    Some(())
}

fn item_hint(
    config: &InlayHintsConfig,
    extern_block: &ast::ExternBlock,
    token: SyntaxToken,
) -> InlayHint {
    InlayHint {
        range: token.text_range(),
        position: crate::InlayHintPosition::Before,
        pad_left: false,
        pad_right: true,
        kind: crate::InlayKind::ExternUnsafety,
        label: crate::InlayHintLabel::from("unsafe"),
        text_edit: Some(config.lazy_text_edit(|| {
            let mut builder = TextEdit::builder();
            builder.insert(token.text_range().start(), "unsafe ".to_owned());
            if extern_block.unsafe_token().is_none() {
                if let Some(abi) = extern_block.abi() {
                    builder.insert(abi.syntax().text_range().start(), "unsafe ".to_owned());
                }
            }
            builder.finish()
        })),
        resolve_parent: Some(extern_block.syntax().text_range()),
    }
}

#[cfg(test)]
mod tests {
    use crate::inlay_hints::tests::{DISABLED_CONFIG, check_with_config};

    #[test]
    fn unadorned() {
        check_with_config(
            DISABLED_CONFIG,
            r#"
  extern "C" {
//^^^^^^^^^^ unsafe
    static FOO: ();
 // ^^^^^^ unsafe
    pub static FOO: ();
     // ^^^^^^unsafe
    unsafe static FOO: ();
    safe static FOO: ();
    fn foo();
 // ^^ unsafe
    pub fn foo();
     // ^^ unsafe
    unsafe fn foo();
    safe fn foo();
}
"#,
        );
    }

    #[test]
    fn adorned() {
        check_with_config(
            DISABLED_CONFIG,
            r#"
unsafe extern "C" {
    static FOO: ();
 // ^^^^^^ unsafe
    pub static FOO: ();
     // ^^^^^^unsafe
    unsafe static FOO: ();
    safe static FOO: ();
    fn foo();
 // ^^ unsafe
    pub fn foo();
     // ^^ unsafe
    unsafe fn foo();
    safe fn foo();
}
"#,
        );
    }
}
