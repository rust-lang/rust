use rustc_hash::{FxHashSet, FxHashMap};

use ra_syntax::{ast, AstNode, TextRange, Direction, SmolStr, SyntaxKind, SyntaxKind::*, SyntaxElement, T};
use ra_db::SourceDatabase;
use ra_prof::profile;

use crate::{FileId, db::RootDatabase};

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
    pub id: Option<u64>,
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

    fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
        use std::{collections::hash_map::DefaultHasher, hash::Hasher};

        let mut hasher = DefaultHasher::new();
        x.hash(&mut hasher);
        hasher.finish()
    }

    // Visited nodes to handle highlighting priorities
    let mut highlighted: FxHashSet<SyntaxElement> = FxHashSet::default();
    let mut bindings_shadow_count: FxHashMap<SmolStr, u32> = FxHashMap::default();

    let mut res = Vec::new();
    for node in source_file.syntax().descendants_with_tokens() {
        if highlighted.contains(&node) {
            continue;
        }
        let (tag, id) = match node.kind() {
            COMMENT => ("comment", None),
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => ("string", None),
            ATTR => ("attribute", None),
            NAME_REF => {
                if let Some(name_ref) = node.as_ast_node::<ast::NameRef>() {
                    use crate::name_ref_kind::{classify_name_ref, NameRefKind::*};
                    use hir::{ModuleDef, ImplItem};

                    // FIXME: try to reuse the SourceAnalyzers
                    let analyzer = hir::SourceAnalyzer::new(db, file_id, name_ref.syntax(), None);
                    match classify_name_ref(db, &analyzer, name_ref) {
                        Some(Method(_)) => ("function", None),
                        Some(Macro(_)) => ("macro", None),
                        Some(FieldAccess(_)) => ("field", None),
                        Some(AssocItem(ImplItem::Method(_))) => ("function", None),
                        Some(AssocItem(ImplItem::Const(_))) => ("constant", None),
                        Some(AssocItem(ImplItem::TypeAlias(_))) => ("type", None),
                        Some(Def(ModuleDef::Module(_))) => ("module", None),
                        Some(Def(ModuleDef::Function(_))) => ("function", None),
                        Some(Def(ModuleDef::Struct(_))) => ("type", None),
                        Some(Def(ModuleDef::Union(_))) => ("type", None),
                        Some(Def(ModuleDef::Enum(_))) => ("type", None),
                        Some(Def(ModuleDef::EnumVariant(_))) => ("constant", None),
                        Some(Def(ModuleDef::Const(_))) => ("constant", None),
                        Some(Def(ModuleDef::Static(_))) => ("constant", None),
                        Some(Def(ModuleDef::Trait(_))) => ("type", None),
                        Some(Def(ModuleDef::TypeAlias(_))) => ("type", None),
                        Some(SelfType(_)) => ("type", None),
                        Some(Pat(ptr)) => ("variable", Some(hash({
                            let text = ptr.syntax_node_ptr().to_node(&source_file.syntax()).text().to_smol_string();
                            let shadow_count = bindings_shadow_count.entry(text.clone()).or_default();
                            (text, shadow_count)
                        }))),
                        Some(SelfParam(_)) => ("type", None),
                        Some(GenericParam(_)) => ("type", None),
                        None => ("text", None),
                    }
                } else {
                    ("text", None)
                }
            }
            NAME => {
                if let Some(name) = node.as_ast_node::<ast::Name>() {
                    ("variable", Some(hash({
                        let text = name.syntax().text().to_smol_string();
                        let shadow_count = bindings_shadow_count.entry(text.clone()).or_insert(1);
                        *shadow_count += 1;
                        (text, shadow_count)
                    })))
                } else {
                    ("text", None)
                }
            }
            TYPE_ALIAS_DEF | TYPE_ARG | TYPE_PARAM => ("type", None),
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => ("literal", None),
            LIFETIME => ("parameter", None),
            T![unsafe] => ("keyword.unsafe", None),
            k if is_control_keyword(k) => ("keyword.control", None),
            k if k.is_keyword() => ("keyword", None),
            _ => {
                // let analyzer = hir::SourceAnalyzer::new(db, file_id, name_ref.syntax(), None);
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
                                    id: None,
                                })
                            }
                        }
                    }
                }
                continue;
            }
        };
        res.push(HighlightedRange { range: node.range(), tag, id })
    }
    res
}

pub(crate) fn highlight_as_html(db: &RootDatabase, file_id: FileId) -> String {
    let source_file = db.parse(file_id);

    let mut ranges = highlight(db, file_id);
    ranges.sort_by_key(|it| it.range.start());
    // quick non-optimal heuristic to intersect token ranges and highlighted ranges
    let mut frontier = 0;
    let mut could_intersect: Vec<&HighlightedRange> = Vec::new();

    let mut buf = String::new();
    buf.push_str(&STYLE);
    buf.push_str("<pre><code>");
    let tokens = source_file.syntax().descendants_with_tokens().filter_map(|it| it.as_token());
    for token in tokens {
        could_intersect.retain(|it| token.range().start() <= it.range.end());
        while let Some(r) = ranges.get(frontier) {
            if r.range.start() <= token.range().end() {
                could_intersect.push(r);
                frontier += 1;
            } else {
                break;
            }
        }
        let text = html_escape(&token.text());
        let classes = could_intersect
            .iter()
            .filter(|it| token.range().is_subrange(&it.range))
            .map(|it| it.tag)
            .collect::<Vec<_>>();
        if classes.is_empty() {
            buf.push_str(&text);
        } else {
            let classes = classes.join(" ");
            buf.push_str(&format!("<span class=\"{}\">{}</span>", classes, text));
        }
    }
    buf.push_str("</code></pre>");
    buf
}

//FIXME: like, real html escaping
fn html_escape(text: &str) -> String {
    text.replace("<", "&lt;").replace(">", "&gt;")
}

const STYLE: &str = "
<style>
pre {
    color: #DCDCCC;
    background-color: #3F3F3F;
    font-size: 22px;
}

.comment   { color: #7F9F7F; }
.string    { color: #CC9393; }
.function  { color: #93E0E3; }
.parameter { color: #94BFF3; }
.builtin   { color: #DD6718; }
.text      { color: #DCDCCC; }
.attribute { color: #BFEBBF; }
.literal   { color: #DFAF8F; }
.macro     { color: #DFAF8F; }

.keyword           { color: #F0DFAF; }
.keyword\\.unsafe  { color: #F0DFAF; font-weight: bold; }
.keyword\\.control { color: #DC8CC3; }

</style>
";

#[cfg(test)]
mod tests {
    use test_utils::{project_dir, read_text, assert_eq_text};
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
fn main() {
    println!("Hello, {}!", 92);

    let mut vec = Vec::new();
    if true {
        vec.push(Foo { x: 0, y: 1 });
    }
    unsafe { vec.set_len(0); }
}
"#,
        );
        let dst_file = project_dir().join("crates/ra_ide_api/src/snapshots/highlighting.html");
        let actual_html = &analysis.highlight_as_html(file_id).unwrap();
        let expected_html = &read_text(&dst_file);
        // std::fs::write(dst_file, &actual_html).unwrap();
        assert_eq_text!(expected_html, actual_html);
    }

    #[test]
    fn test_rainbow_highlighting() {
        let (analysis, file_id) = single_file(
            r#"
fn main() {
    let hello = "hello";
    let x = hello.to_string();
    let y = hello.to_string();

    let x = "other color please!";
    let y = x.to_string();
}"#,
        );
        let result = analysis.highlight(file_id);
        assert_debug_snapshot_matches!("rainbow_highlighting", result);
    }
}
