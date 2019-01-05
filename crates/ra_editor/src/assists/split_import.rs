use ra_syntax::{
    TextUnit, AstNode, SyntaxKind::COLONCOLON,
    ast,
    algo::generate,
};

use crate::assists::{AssistCtx, Assist};

pub fn split_import(ctx: AssistCtx) -> Option<Assist> {
    let colon_colon = ctx
        .leaf_at_offset()
        .find(|leaf| leaf.kind() == COLONCOLON)?;
    let path = colon_colon.parent().and_then(ast::Path::cast)?;
    let top_path = generate(Some(path), |it| it.parent_path()).last()?;

    let use_tree = top_path.syntax().ancestors().find_map(ast::UseTree::cast);
    if use_tree.is_none() {
        return None;
    }

    let l_curly = colon_colon.range().end();
    let r_curly = top_path.syntax().range().end();

    ctx.build("split import", |edit| {
        edit.insert(l_curly, "{");
        edit.insert(r_curly, "}");
        edit.set_cursor(l_curly + TextUnit::of_str("{"));
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assists::check_assist;

    #[test]
    fn test_split_import() {
        check_assist(
            split_import,
            "use crate::<|>db::RootDatabase;",
            "use crate::{<|>db::RootDatabase};",
        )
    }
}
