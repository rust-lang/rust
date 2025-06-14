//! Implementation of "closing brace" inlay hints:
//! ```no_run
//! fn g() {
//! } /* fn g */
//! ```
use hir::{DisplayTarget, HirDisplay, InRealFile, Semantics};
use ide_db::{FileRange, RootDatabase};
use syntax::{
    SyntaxKind, SyntaxNode, T,
    ast::{self, AstNode, HasLoopBody, HasName},
    match_ast,
};

use crate::{
    InlayHint, InlayHintLabel, InlayHintPosition, InlayHintsConfig, InlayKind,
    inlay_hints::LazyProperty,
};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<'_, RootDatabase>,
    config: &InlayHintsConfig,
    display_target: DisplayTarget,
    InRealFile { file_id, value: node }: InRealFile<SyntaxNode>,
) -> Option<()> {
    let min_lines = config.closing_brace_hints_min_lines?;

    let name = |it: ast::Name| it.syntax().text_range();

    let mut node = node.clone();
    let mut closing_token;
    let (label, name_range) = if let Some(item_list) = ast::AssocItemList::cast(node.clone()) {
        closing_token = item_list.r_curly_token()?;

        let parent = item_list.syntax().parent()?;
        match_ast! {
            match parent {
                ast::Impl(imp) => {
                    let imp = sema.to_def(&imp)?;
                    let ty = imp.self_ty(sema.db);
                    let trait_ = imp.trait_(sema.db);
                    let hint_text = match trait_ {
                        Some(tr) => format!(
                            "impl {} for {}",
                            tr.name(sema.db).display(sema.db, display_target.edition),
                            ty.display_truncated(sema.db, config.max_length, display_target,
                        )),
                        None => format!("impl {}", ty.display_truncated(sema.db, config.max_length, display_target)),
                    };
                    (hint_text, None)
                },
                ast::Trait(tr) => {
                    (format!("trait {}", tr.name()?), tr.name().map(name))
                },
                _ => return None,
            }
        }
    } else if let Some(list) = ast::ItemList::cast(node.clone()) {
        closing_token = list.r_curly_token()?;

        let module = ast::Module::cast(list.syntax().parent()?)?;
        (format!("mod {}", module.name()?), module.name().map(name))
    } else if let Some(label) = ast::Label::cast(node.clone()) {
        // in this case, `ast::Label` could be seen as a part of `ast::BlockExpr`
        // the actual number of lines in this case should be the line count of the parent BlockExpr,
        // which the `min_lines` config cares about
        node = node.parent()?;

        let parent = label.syntax().parent()?;
        let block;
        match_ast! {
            match parent {
                ast::BlockExpr(block_expr) => {
                    block = block_expr.stmt_list()?;
                },
                ast::AnyHasLoopBody(loop_expr) => {
                    block = loop_expr.loop_body()?.stmt_list()?;
                },
                _ => return None,
            }
        }
        closing_token = block.r_curly_token()?;

        let lifetime = label.lifetime()?.to_string();

        (lifetime, Some(label.syntax().text_range()))
    } else if let Some(block) = ast::BlockExpr::cast(node.clone()) {
        closing_token = block.stmt_list()?.r_curly_token()?;

        let parent = block.syntax().parent()?;
        match_ast! {
            match parent {
                ast::Fn(it) => {
                    // FIXME: this could include parameters, but `HirDisplay` prints too much info
                    // and doesn't respect the max length either, so the hints end up way too long
                    (format!("fn {}", it.name()?), it.name().map(name))
                },
                ast::Static(it) => (format!("static {}", it.name()?), it.name().map(name)),
                ast::Const(it) => {
                    if it.underscore_token().is_some() {
                        ("const _".into(), None)
                    } else {
                        (format!("const {}", it.name()?), it.name().map(name))
                    }
                },
                _ => return None,
            }
        }
    } else if let Some(mac) = ast::MacroCall::cast(node.clone()) {
        let last_token = mac.syntax().last_token()?;
        if last_token.kind() != T![;] && last_token.kind() != SyntaxKind::R_CURLY {
            return None;
        }
        closing_token = last_token;

        (
            format!("{}!", mac.path()?),
            mac.path().and_then(|it| it.segment()).map(|it| it.syntax().text_range()),
        )
    } else {
        return None;
    };

    if let Some(mut next) = closing_token.next_token() {
        if next.kind() == T![;] {
            if let Some(tok) = next.next_token() {
                closing_token = next;
                next = tok;
            }
        }
        if !(next.kind() == SyntaxKind::WHITESPACE && next.text().contains('\n')) {
            // Only display the hint if the `}` is the last token on the line
            return None;
        }
    }

    let mut lines = 1;
    node.text().for_each_chunk(|s| lines += s.matches('\n').count());
    if lines < min_lines {
        return None;
    }

    let linked_location =
        name_range.map(|range| FileRange { file_id: file_id.file_id(sema.db), range });
    acc.push(InlayHint {
        range: closing_token.text_range(),
        kind: InlayKind::ClosingBrace,
        label: InlayHintLabel::simple(label, None, linked_location.map(LazyProperty::Computed)),
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: true,
        pad_right: false,
        resolve_parent: Some(node.text_range()),
    });

    None
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig,
        inlay_hints::tests::{DISABLED_CONFIG, check_with_config},
    };

    #[test]
    fn hints_closing_brace() {
        check_with_config(
            InlayHintsConfig { closing_brace_hints_min_lines: Some(2), ..DISABLED_CONFIG },
            r#"
fn a() {}

fn f() {
} // no hint unless `}` is the last token on the line

fn g() {
  }
//^ fn g

fn h<T>(with: T, arguments: u8, ...) {
  }
//^ fn h

trait Tr {
    fn f();
    fn g() {
    }
  //^ fn g
  }
//^ trait Tr
impl Tr for () {
  }
//^ impl Tr for ()
impl dyn Tr {
  }
//^ impl dyn Tr + 'static

static S0: () = 0;
static S1: () = {};
static S2: () = {
 };
//^ static S2
const _: () = {
 };
//^ const _

mod m {
  }
//^ mod m

m! {}
m!();
m!(
 );
//^ m!

m! {
  }
//^ m!

fn f() {
    let v = vec![
    ];
  }
//^ fn f
"#,
        );
    }

    #[test]
    fn hints_closing_brace_for_block_expr() {
        check_with_config(
            InlayHintsConfig { closing_brace_hints_min_lines: Some(2), ..DISABLED_CONFIG },
            r#"
fn test() {
    'end: {
        'do_a: {
            'do_b: {

            }
          //^ 'do_b
            break 'end;
        }
      //^ 'do_a
    }
  //^ 'end

    'a: loop {
        'b: for i in 0..5 {
            'c: while true {


            }
          //^ 'c
        }
      //^ 'b
    }
  //^ 'a

  }
//^ fn test
"#,
        );
    }
}
