use rustc_hash::FxHashSet;

use ra_syntax::{ast, AstNode, TextRange, Direction, SyntaxKind, SyntaxKind::*, SyntaxElement, T};
use ra_db::SourceDatabase;
use ra_prof::profile;

use crate::{FileId, db::RootDatabase};

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

fn is_control_keyword(kind: SyntaxKind) -> bool {
    match kind {
        T![for]
        | T![loop]
        | T![while]
        | T![continue]
        | T![break]
        | T![if]
        | T![else]
        | T![match]
        | T![return] => true,
        _ => false,
    }
}

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Vec<HighlightedRange> {
    let _p = profile("highlight");

    let source_file = db.parse(file_id);

    // Visited nodes to handle highlighting priorities
    let mut highlighted: FxHashSet<SyntaxElement> = FxHashSet::default();
    let mut res = Vec::new();
    for node in source_file.syntax().descendants_with_tokens() {
        if highlighted.contains(&node) {
            continue;
        }
        let tag = match node.kind() {
            COMMENT => "comment",
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => "string",
            ATTR => "attribute",
            NAME_REF => {
                if let Some(name_ref) = node.as_node().and_then(|n| ast::NameRef::cast(n)) {
                    use crate::name_ref_kind::{classify_name_ref, NameRefKind::*};
                    use hir::{ModuleDef, ImplItem};

                    // FIXME: try to reuse the SourceAnalyzers
                    let analyzer = hir::SourceAnalyzer::new(db, file_id, name_ref.syntax(), None);
                    match classify_name_ref(db, &analyzer, name_ref) {
                        Some(Method(_)) => "function",
                        Some(Macro(_)) => "macro",
                        Some(FieldAccess(_)) => "field",
                        Some(AssocItem(ImplItem::Method(_))) => "function",
                        Some(AssocItem(ImplItem::Const(_))) => "constant",
                        Some(AssocItem(ImplItem::TypeAlias(_))) => "type",
                        Some(Def(ModuleDef::Module(_))) => "module",
                        Some(Def(ModuleDef::Function(_))) => "function",
                        Some(Def(ModuleDef::Struct(_))) => "type",
                        Some(Def(ModuleDef::Union(_))) => "type",
                        Some(Def(ModuleDef::Enum(_))) => "type",
                        Some(Def(ModuleDef::EnumVariant(_))) => "constant",
                        Some(Def(ModuleDef::Const(_))) => "constant",
                        Some(Def(ModuleDef::Static(_))) => "constant",
                        Some(Def(ModuleDef::Trait(_))) => "type",
                        Some(Def(ModuleDef::TypeAlias(_))) => "type",
                        Some(SelfType(_)) => "type",
                        Some(Pat(_)) => "text",
                        Some(SelfParam(_)) => "type",
                        Some(GenericParam(_)) => "type",
                        None => "text",
                    }
                } else {
                    "text"
                }
            }
            NAME => "function",
            TYPE_ALIAS_DEF | TYPE_ARG | TYPE_PARAM => "type",
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => "literal",
            LIFETIME => "parameter",
            T![unsafe] => "keyword.unsafe",
            k if is_control_keyword(k) => "keyword.control",
            k if k.is_keyword() => "keyword",
            _ => {
                if let Some(macro_call) = node.as_node().and_then(ast::MacroCall::cast) {
                    if let Some(path) = macro_call.path() {
                        if let Some(segment) = path.segment() {
                            if let Some(name_ref) = segment.name_ref() {
                                highlighted.insert(name_ref.syntax().into());
                                let range_start = name_ref.syntax().range().start();
                                let mut range_end = name_ref.syntax().range().end();
                                for sibling in path.syntax().siblings_with_tokens(Direction::Next) {
                                    match sibling.kind() {
                                        T![!] | IDENT => range_end = sibling.range().end(),
                                        _ => (),
                                    }
                                }
                                res.push(HighlightedRange {
                                    range: TextRange::from_to(range_start, range_end),
                                    tag: "macro",
                                })
                            }
                        }
                    }
                }
                continue;
            }
        };
        res.push(HighlightedRange { range: node.range(), tag })
    }
    res
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;

    use crate::mock_analysis::single_file;

    #[test]
    fn test_highlighting() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
}

fn foo<T>() -> T {
    unimplemented!();
}

// comment
fn main() {}
    println!("Hello, {}!", 92);

    let mut vec = Vec::new();
    vec.push(Foo { x: 0, y: 1 });
    unsafe { vec.set_len(0); }
"#,
        );
        let result = analysis.highlight(file_id);
        assert_debug_snapshot_matches!("highlighting", result);
    }
}
