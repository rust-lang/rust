//! Extern block hints
use ide_db::{famous_defs::FamousDefs, text_edit::TextEdit};
use span::EditionedFileId;
use syntax::{ast, AstNode, SyntaxToken};

use crate::{InlayHint, InlayHintsConfig};

pub(super) fn extern_block_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(_sema, _): &FamousDefs<'_, '_>,
    _config: &InlayHintsConfig,
    _file_id: EditionedFileId,
    extern_block: ast::ExternBlock,
) -> Option<()> {
    if extern_block.unsafe_token().is_some() {
        return None;
    }
    let abi = extern_block.abi()?;
    acc.push(InlayHint {
        range: abi.syntax().text_range(),
        position: crate::InlayHintPosition::Before,
        pad_left: false,
        pad_right: true,
        kind: crate::InlayKind::ExternUnsafety,
        label: crate::InlayHintLabel::from("unsafe"),
        text_edit: Some(TextEdit::insert(abi.syntax().text_range().start(), "unsafe ".to_owned())),
        resolve_parent: Some(extern_block.syntax().text_range()),
    });
    Some(())
}

pub(super) fn fn_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(_sema, _): &FamousDefs<'_, '_>,
    _config: &InlayHintsConfig,
    _file_id: EditionedFileId,
    fn_: &ast::Fn,
    extern_block: &ast::ExternBlock,
) -> Option<()> {
    let implicit_unsafe = fn_.safe_token().is_none() && fn_.unsafe_token().is_none();
    if !implicit_unsafe {
        return None;
    }
    let fn_ = fn_.fn_token()?;
    acc.push(item_hint(extern_block, fn_));
    Some(())
}

pub(super) fn static_hints(
    acc: &mut Vec<InlayHint>,
    FamousDefs(_sema, _): &FamousDefs<'_, '_>,
    _config: &InlayHintsConfig,
    _file_id: EditionedFileId,
    static_: &ast::Static,
    extern_block: &ast::ExternBlock,
) -> Option<()> {
    let implicit_unsafe = static_.safe_token().is_none() && static_.unsafe_token().is_none();
    if !implicit_unsafe {
        return None;
    }
    let static_ = static_.static_token()?;
    acc.push(item_hint(extern_block, static_));
    Some(())
}

fn item_hint(extern_block: &ast::ExternBlock, token: SyntaxToken) -> InlayHint {
    InlayHint {
        range: token.text_range(),
        position: crate::InlayHintPosition::Before,
        pad_left: false,
        pad_right: true,
        kind: crate::InlayKind::ExternUnsafety,
        label: crate::InlayHintLabel::from("unsafe"),
        text_edit: {
            let mut builder = TextEdit::builder();
            builder.insert(token.text_range().start(), "unsafe ".to_owned());
            if extern_block.unsafe_token().is_none() {
                if let Some(abi) = extern_block.abi() {
                    builder.insert(abi.syntax().text_range().start(), "unsafe ".to_owned());
                }
            }
            Some(builder.finish())
        },
        resolve_parent: Some(extern_block.syntax().text_range()),
    }
}

#[cfg(test)]
mod tests {
    use crate::inlay_hints::tests::{check_with_config, DISABLED_CONFIG};

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
