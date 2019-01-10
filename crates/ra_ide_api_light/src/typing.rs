use ra_syntax::{
    algo::{find_node_at_offset, find_leaf_at_offset, LeafAtOffset},
    ast,
    AstNode, Direction, SourceFile, SyntaxKind::*,
    SyntaxNode, TextUnit,
};

use crate::{LocalEdit, TextEditBuilder};

pub fn on_enter(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let comment = find_leaf_at_offset(file.syntax(), offset)
        .left_biased()
        .and_then(ast::Comment::cast)?;

    if let ast::CommentFlavor::Multiline = comment.flavor() {
        return None;
    }

    let prefix = comment.prefix();
    if offset < comment.syntax().range().start() + TextUnit::of_str(prefix) + TextUnit::from(1) {
        return None;
    }

    let indent = node_indent(file, comment.syntax())?;
    let inserted = format!("\n{}{} ", indent, prefix);
    let cursor_position = offset + TextUnit::of_str(&inserted);
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, inserted);
    Some(LocalEdit {
        label: "on enter".to_string(),
        edit: edit.finish(),
        cursor_position: Some(cursor_position),
    })
}

fn node_indent<'a>(file: &'a SourceFile, node: &SyntaxNode) -> Option<&'a str> {
    let ws = match find_leaf_at_offset(file.syntax(), node.range().start()) {
        LeafAtOffset::Between(l, r) => {
            assert!(r == node);
            l
        }
        LeafAtOffset::Single(n) => {
            assert!(n == node);
            return Some("");
        }
        LeafAtOffset::None => unreachable!(),
    };
    if ws.kind() != WHITESPACE {
        return None;
    }
    let text = ws.leaf_text().unwrap();
    let pos = text.as_str().rfind('\n').map(|it| it + 1).unwrap_or(0);
    Some(&text[pos..])
}

pub fn on_eq_typed(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let let_stmt: &ast::LetStmt = find_node_at_offset(file.syntax(), offset)?;
    if let_stmt.has_semi() {
        return None;
    }
    if let Some(expr) = let_stmt.initializer() {
        let expr_range = expr.syntax().range();
        if expr_range.contains(offset) && offset != expr_range.start() {
            return None;
        }
        if file
            .syntax()
            .text()
            .slice(offset..expr_range.start())
            .contains('\n')
        {
            return None;
        }
    } else {
        return None;
    }
    let offset = let_stmt.syntax().range().end();
    let mut edit = TextEditBuilder::default();
    edit.insert(offset, ";".to_string());
    Some(LocalEdit {
        label: "add semicolon".to_string(),
        edit: edit.finish(),
        cursor_position: None,
    })
}

pub fn on_dot_typed(file: &SourceFile, offset: TextUnit) -> Option<LocalEdit> {
    let before_dot_offset = offset - TextUnit::of_char('.');

    let whitespace = find_leaf_at_offset(file.syntax(), before_dot_offset).left_biased()?;

    // find whitespace just left of the dot
    ast::Whitespace::cast(whitespace)?;

    // make sure there is a method call
    let method_call = whitespace
        .siblings(Direction::Prev)
        // first is whitespace
        .skip(1)
        .next()?;

    ast::MethodCallExpr::cast(method_call)?;

    // find how much the _method call is indented
    let method_chain_indent = method_call
        .parent()?
        .siblings(Direction::Prev)
        .skip(1)
        .next()?
        .leaf_text()
        .map(|x| last_line_indent_in_whitespace(x))?;

    let current_indent = TextUnit::of_str(last_line_indent_in_whitespace(whitespace.leaf_text()?));
    // TODO: indent is always 4 spaces now. A better heuristic could look on the previous line(s)

    let target_indent = TextUnit::of_str(method_chain_indent) + TextUnit::from_usize(4);

    let diff = target_indent - current_indent;

    let indent = "".repeat(diff.to_usize());

    let cursor_position = offset + diff;
    let mut edit = TextEditBuilder::default();
    edit.insert(before_dot_offset, indent);
    Some(LocalEdit {
        label: "indent dot".to_string(),
        edit: edit.finish(),
        cursor_position: Some(cursor_position),
    })
}

/// Finds the last line in the whitespace
fn last_line_indent_in_whitespace(ws: &str) -> &str {
    ws.split('\n').last().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{add_cursor, assert_eq_text, extract_offset};

    use super::*;

    #[test]
    fn test_on_eq_typed() {
        fn do_check(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            let result = on_eq_typed(&file, offset).unwrap();
            let actual = result.edit.apply(&before);
            assert_eq_text!(after, &actual);
        }

        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        // }
        // ");
        do_check(
            r"
fn foo() {
    let foo =<|> 1 + 1
}
",
            r"
fn foo() {
    let foo = 1 + 1;
}
",
        );
        //     do_check(r"
        // fn foo() {
        //     let foo =<|>
        //     let bar = 1;
        // }
        // ", r"
        // fn foo() {
        //     let foo =;
        //     let bar = 1;
        // }
        // ");
    }

    #[test]
    fn test_on_dot_typed() {
        fn do_check(before: &str, after: &str) {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            if let Some(result) = on_eq_typed(&file, offset) {
                let actual = result.edit.apply(&before);
                assert_eq_text!(after, &actual);
            };
        }
        // indent if continuing chain call
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );

        // do not indent if already indented
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );

        // indent if the previous line is already indented
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .
    }
",
        );

        // don't indent if indent matches previous line
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .first()
            .
    }
",
        );

        // don't indent if there is no method call on previous line
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        .<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        .
    }
",
        );

        // indent to match previous expr
        do_check(
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
.<|>
    }
",
            r"
    pub fn child(&self, db: &impl HirDatabase, name: &Name) -> Cancelable<Option<Module>> {
        self.child_impl(db, name)
            .
    }
",
        );
    }

    #[test]
    fn test_on_enter() {
        fn apply_on_enter(before: &str) -> Option<String> {
            let (offset, before) = extract_offset(before);
            let file = SourceFile::parse(&before);
            let result = on_enter(&file, offset)?;
            let actual = result.edit.apply(&before);
            let actual = add_cursor(&actual, result.cursor_position.unwrap());
            Some(actual)
        }

        fn do_check(before: &str, after: &str) {
            let actual = apply_on_enter(before).unwrap();
            assert_eq_text!(after, &actual);
        }

        fn do_check_noop(text: &str) {
            assert!(apply_on_enter(text).is_none())
        }

        do_check(
            r"
/// Some docs<|>
fn foo() {
}
",
            r"
/// Some docs
/// <|>
fn foo() {
}
",
        );
        do_check(
            r"
impl S {
    /// Some<|> docs.
    fn foo() {}
}
",
            r"
impl S {
    /// Some
    /// <|> docs.
    fn foo() {}
}
",
        );
        do_check_noop(r"<|>//! docz");
    }
}
