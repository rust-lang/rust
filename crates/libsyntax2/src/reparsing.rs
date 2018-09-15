use algo;
use grammar;
use lexer::{tokenize, Token};
use text_unit::{TextRange, TextUnit};
use yellow::{self, SyntaxNodeRef, GreenNode, SyntaxError};
use parser_impl;
use parser_api::Parser;
use {
    SyntaxKind::*,
};
use text_utils::replace_range;

#[derive(Debug, Clone)]
pub struct AtomEdit {
    pub delete: TextRange,
    pub insert: String,
}

impl AtomEdit {
    pub fn replace(range: TextRange, replace_with: String) -> AtomEdit {
        AtomEdit { delete: range, insert: replace_with }
    }

    pub fn delete(range: TextRange) -> AtomEdit {
        AtomEdit::replace(range, String::new())
    }

    pub fn insert(offset: TextUnit, text: String) -> AtomEdit {
        AtomEdit::replace(TextRange::offset_len(offset, 0.into()), text)
    }
}

pub(crate) fn incremental_reparse(
    node: SyntaxNodeRef,
    edit: &AtomEdit,
    errors: Vec<SyntaxError>,
) -> Option<(GreenNode, Vec<SyntaxError>)> {
    let (node, green, new_errors) =
        reparse_leaf(node, &edit).or_else(|| reparse_block(node, &edit))?;
    let green_root = node.replace_with(green);
    let errors = merge_errors(errors, new_errors, node, edit);
    Some((green_root, errors))
}

fn reparse_leaf<'node>(
    node: SyntaxNodeRef<'node>,
    edit: &AtomEdit,
) -> Option<(SyntaxNodeRef<'node>, GreenNode, Vec<SyntaxError>)> {
    let node = algo::find_covering_node(node, edit.delete);
    match node.kind() {
        | WHITESPACE
        | COMMENT
        | DOC_COMMENT
        | IDENT
        | STRING
        | RAW_STRING => {
            let text = get_text_after_edit(node, &edit);
            let tokens = tokenize(&text);
            let token = match tokens[..] {
                [token] if token.kind == node.kind() => token,
                _ => return None,
            };

            if token.kind == IDENT && is_contextual_kw(&text) {
                return None;
            }

            let green = GreenNode::new_leaf(node.kind(), &text);
            let new_errors = vec![];
            Some((node, green, new_errors))
        }
        _ => None,
    }
}

fn reparse_block<'node>(
    node: SyntaxNodeRef<'node>,
    edit: &AtomEdit,
) -> Option<(SyntaxNodeRef<'node>, GreenNode, Vec<SyntaxError>)> {
    let (node, reparser) = find_reparsable_node(node, edit.delete)?;
    let text = get_text_after_edit(node, &edit);
    let tokens = tokenize(&text);
    if !is_balanced(&tokens) {
        return None;
    }
    let (green, new_errors) =
        parser_impl::parse_with::<yellow::GreenBuilder>(
            &text, &tokens, reparser,
        );
    Some((node, green, new_errors))
}

fn get_text_after_edit(node: SyntaxNodeRef, edit: &AtomEdit) -> String {
    replace_range(
        node.text().to_string(),
        edit.delete - node.range().start(),
        &edit.insert,
    )
}

fn is_contextual_kw(text: &str) -> bool {
    match text {
        | "auto"
        | "default"
        | "union" => true,
        _ => false,
    }
}

fn find_reparsable_node<'node>(
    node: SyntaxNodeRef<'node>,
    range: TextRange,
) -> Option<(SyntaxNodeRef<'node>, fn(&mut Parser))> {
    let node = algo::find_covering_node(node, range);
    return algo::ancestors(node)
        .filter_map(|node| reparser(node).map(|r| (node, r)))
        .next();

    fn reparser(node: SyntaxNodeRef) -> Option<fn(&mut Parser)> {
        let res = match node.kind() {
            BLOCK => grammar::block,
            NAMED_FIELD_DEF_LIST => grammar::named_field_def_list,
            NAMED_FIELD_LIST => grammar::named_field_list,
            ENUM_VARIANT_LIST => grammar::enum_variant_list,
            MATCH_ARM_LIST => grammar::match_arm_list,
            USE_TREE_LIST => grammar::use_tree_list,
            EXTERN_ITEM_LIST => grammar::extern_item_list,
            TOKEN_TREE if node.first_child().unwrap().kind() == L_CURLY => grammar::token_tree,
            ITEM_LIST => {
                let parent = node.parent().unwrap();
                match parent.kind() {
                    IMPL_ITEM => grammar::impl_item_list,
                    TRAIT_DEF => grammar::trait_item_list,
                    MODULE => grammar::mod_item_list,
                    _ => return None,
                }
            }
            _ => return None,
        };
        Some(res)
    }
}

fn is_balanced(tokens: &[Token]) -> bool {
    if tokens.len() == 0
        || tokens.first().unwrap().kind != L_CURLY
        || tokens.last().unwrap().kind != R_CURLY {
        return false;
    }
    let mut balance = 0usize;
    for t in tokens.iter() {
        match t.kind {
            L_CURLY => balance += 1,
            R_CURLY => balance = match balance.checked_sub(1) {
                Some(b) => b,
                None => return false,
            },
            _ => (),
        }
    }
    balance == 0
}

fn merge_errors(
    old_errors: Vec<SyntaxError>,
    new_errors: Vec<SyntaxError>,
    old_node: SyntaxNodeRef,
    edit: &AtomEdit,
) -> Vec<SyntaxError> {
    let mut res = Vec::new();
    for e in old_errors {
        if e.offset <= old_node.range().start() {
            res.push(e)
        } else if e.offset >= old_node.range().end() {
            res.push(SyntaxError {
                msg: e.msg,
                offset: e.offset + TextUnit::of_str(&edit.insert) - edit.delete.len(),
            })
        }
    }
    for e in new_errors {
        res.push(SyntaxError {
            msg: e.msg,
            offset: e.offset + old_node.range().start(),
        })
    }
    res
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            File,
            test_utils::extract_range,
            text_utils::replace_range,
            utils::dump_tree,
        },
        reparse_leaf, reparse_block, AtomEdit, GreenNode, SyntaxError, SyntaxNodeRef,
    };

    fn do_check<F>(
        before: &str,
        replace_with: &str,
        reparser: F,
    ) where
        for<'a> F: Fn(
            SyntaxNodeRef<'a>,
            &AtomEdit,
        ) -> Option<(SyntaxNodeRef<'a>, GreenNode, Vec<SyntaxError>)>
    {
        let (range, before) = extract_range(before);
        let after = replace_range(before.clone(), range, replace_with);

        let fully_reparsed = File::parse(&after);
        let incrementally_reparsed = {
            let f = File::parse(&before);
            let edit = AtomEdit { delete: range, insert: replace_with.to_string() };
            let (node, green, new_errors) =
                reparser(f.syntax(), &edit).expect("cannot incrementally reparse");
            let green_root = node.replace_with(green);
            let errors = super::merge_errors(f.errors(), new_errors, node, &edit);
            File::new(green_root, errors)
        };

        assert_eq_text!(
            &dump_tree(fully_reparsed.syntax()),
            &dump_tree(incrementally_reparsed.syntax()),
        )
    }

    #[test]
    fn reparse_block_tests() {
        let do_check = |before, replace_to|
            do_check(before, replace_to, reparse_block);

        do_check(r"
fn foo() {
    let x = foo + <|>bar<|>
}
", "baz");
        do_check(r"
fn foo() {
    let x = foo<|> + bar<|>
}
", "baz");
        do_check(r"
struct Foo {
    f: foo<|><|>
}
", ",\n    g: (),");
        do_check(r"
fn foo {
    let;
    1 + 1;
    <|>92<|>;
}
", "62");
        do_check(r"
mod foo {
    fn <|><|>
}
", "bar");
        do_check(r"
trait Foo {
    type <|>Foo<|>;
}
", "Output");
        do_check(r"
impl IntoIterator<Item=i32> for Foo {
    f<|><|>
}
", "n next(");
        do_check(r"
use a::b::{foo,<|>,bar<|>};
    ", "baz");
        do_check(r"
pub enum A {
    Foo<|><|>
}
", "\nBar;\n");
        do_check(r"
foo!{a, b<|><|> d}
", ", c[3]");
        do_check(r"
fn foo() {
    vec![<|><|>]
}
", "123");
        do_check(r"
extern {
    fn<|>;<|>
}
", " exit(code: c_int)");
    }

    #[test]
    fn reparse_leaf_tests() {
        let do_check = |before, replace_to|
            do_check(before, replace_to, reparse_leaf);

        do_check(r"<|><|>
fn foo() -> i32 { 1 }
", "\n\n\n   \n");
        do_check(r"
fn foo() -> <|><|> {}
", "  \n");
        do_check(r"
fn <|>foo<|>() -> i32 { 1 }
", "bar");
        do_check(r"
fn foo<|><|>foo() {  }
", "bar");
        do_check(r"
fn foo /* <|><|> */ () {}
", "some comment");
        do_check(r"
fn baz <|><|> () {}
", "    \t\t\n\n");
        do_check(r"
fn baz <|><|> () {}
", "    \t\t\n\n");
        do_check(r"
/// foo <|><|>omment
mod { }
", "c");
        do_check(r#"
fn -> &str { "Hello<|><|>" }
"#, ", world");
        do_check(r#"
fn -> &str { // "Hello<|><|>"
"#, ", world");
        do_check(r##"
fn -> &str { r#"Hello<|><|>"#
"##, ", world");
        do_check(r"
#[derive(<|>Copy<|>)]
enum Foo {

}
", "Clone");
    }
}