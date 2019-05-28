use rustc_hash::{FxHashSet, FxHashMap};

use ra_syntax::{ast, AstNode, TextRange, Direction, SmolStr, SyntaxKind, SyntaxKind::*, SyntaxElement, T};
use ra_db::SourceDatabase;
use ra_prof::profile;

use crate::{FileId, db::RootDatabase};

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
    pub binding_hash: Option<u64>,
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
    let source_file = db.parse(file_id).tree;

    fn calc_binding_hash(file_id: FileId, text: &SmolStr, shadow_count: u32) -> u64 {
        fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
            use std::{collections::hash_map::DefaultHasher, hash::Hasher};

            let mut hasher = DefaultHasher::new();
            x.hash(&mut hasher);
            hasher.finish()
        }

        hash((file_id, text, shadow_count))
    }

    // Visited nodes to handle highlighting priorities
    let mut highlighted: FxHashSet<SyntaxElement> = FxHashSet::default();
    let mut bindings_shadow_count: FxHashMap<SmolStr, u32> = FxHashMap::default();

    let mut res = Vec::new();
    for node in source_file.syntax().descendants_with_tokens() {
        if highlighted.contains(&node) {
            continue;
        }
        let mut binding_hash = None;
        let tag = match node.kind() {
            COMMENT => "comment",
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => "string",
            ATTR => "attribute",
            NAME_REF => {
                if let Some(name_ref) = node.as_node().and_then(ast::NameRef::cast) {
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
                        Some(Pat(ptr)) => {
                            binding_hash = Some({
                                let text = ptr
                                    .syntax_node_ptr()
                                    .to_node(&source_file.syntax())
                                    .text()
                                    .to_smol_string();
                                let shadow_count =
                                    bindings_shadow_count.entry(text.clone()).or_default();
                                calc_binding_hash(file_id, &text, *shadow_count)
                            });

                            "variable"
                        }
                        Some(SelfParam(_)) => "type",
                        Some(GenericParam(_)) => "type",
                        None => "text",
                    }
                } else {
                    "text"
                }
            }
            NAME => {
                if let Some(name) = node.as_node().and_then(ast::Name::cast) {
                    if name.syntax().ancestors().any(|x| ast::BindPat::cast(x).is_some()) {
                        binding_hash = Some({
                            let text = name.syntax().text().to_smol_string();
                            let shadow_count =
                                bindings_shadow_count.entry(text.clone()).or_insert(0);
                            *shadow_count += 1;
                            calc_binding_hash(file_id, &text, *shadow_count)
                        });
                        "variable"
                    } else {
                        "function"
                    }
                } else {
                    "text"
                }
            }
            TYPE_ALIAS_DEF | TYPE_ARG | TYPE_PARAM => "type",
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => "literal",
            LIFETIME => "parameter",
            T![unsafe] => "keyword.unsafe",
            k if is_control_keyword(k) => "keyword.control",
            k if k.is_keyword() => "keyword",
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
                                    binding_hash: None,
                                })
                            }
                        }
                    }
                }
                continue;
            }
        };
        res.push(HighlightedRange { range: node.range(), tag, binding_hash })
    }
    res
}

pub(crate) fn highlight_as_html(db: &RootDatabase, file_id: FileId, rainbow: bool) -> String {
    let source_file = db.parse(file_id).tree;

    fn rainbowify(seed: u64) -> String {
        use rand::prelude::*;
        let mut rng = SmallRng::seed_from_u64(seed);
        format!(
            "hsl({h},{s}%,{l}%)",
            h = rng.gen_range::<u16, _, _>(0, 361),
            s = rng.gen_range::<u16, _, _>(42, 99),
            l = rng.gen_range::<u16, _, _>(40, 91),
        )
    }

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
        let ranges = could_intersect
            .iter()
            .filter(|it| token.range().is_subrange(&it.range))
            .collect::<Vec<_>>();
        if ranges.is_empty() {
            buf.push_str(&text);
        } else {
            let classes = ranges.iter().map(|x| x.tag).collect::<Vec<_>>().join(" ");
            let binding_hash = ranges.first().and_then(|x| x.binding_hash);
            let color = match (rainbow, binding_hash) {
                (true, Some(hash)) => format!(
                    " data-binding-hash=\"{}\" style=\"color: {};\"",
                    hash,
                    rainbowify(hash)
                ),
                _ => "".into(),
            };
            buf.push_str(&format!("<span class=\"{}\"{}>{}</span>", classes, color, text));
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
body       { margin: 0; }
pre        { color: #DCDCCC; background: #3F3F3F; font-size: 22px; padding: 0.4em; }

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
"#
            .trim(),
        );
        let dst_file = project_dir().join("crates/ra_ide_api/src/snapshots/highlighting.html");
        let actual_html = &analysis.highlight_as_html(file_id, true).unwrap();
        let expected_html = &read_text(&dst_file);
        std::fs::write(dst_file, &actual_html).unwrap();
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
}
"#
            .trim(),
        );
        let dst_file =
            project_dir().join("crates/ra_ide_api/src/snapshots/rainbow_highlighting.html");
        let actual_html = &analysis.highlight_as_html(file_id, true).unwrap();
        let expected_html = &read_text(&dst_file);
        std::fs::write(dst_file, &actual_html).unwrap();
        assert_eq_text!(expected_html, actual_html);
    }
}
