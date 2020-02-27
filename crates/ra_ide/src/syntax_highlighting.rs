//! FIXME: write short doc here

mod tags;
mod html;

use hir::{Name, Semantics};
use ra_ide_db::{
    defs::{classify_name, NameDefinition},
    RootDatabase,
};
use ra_prof::profile;
use ra_syntax::{
    ast, AstNode, Direction, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxKind::*, SyntaxToken,
    TextRange, WalkEvent, T,
};
use rustc_hash::FxHashMap;

use crate::{references::classify_name_ref, FileId};

pub use tags::{Highlight, HighlightModifier, HighlightModifiers, HighlightTag};

pub(crate) use html::highlight_as_html;

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub highlight: Highlight,
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

pub(crate) fn highlight(
    db: &RootDatabase,
    file_id: FileId,
    range_to_highlight: Option<TextRange>,
) -> Vec<HighlightedRange> {
    let _p = profile("highlight");
    let sema = Semantics::new(db);

    // Determine the root based on the given range.
    let (root, range_to_highlight) = {
        let source_file = sema.parse(file_id);
        match range_to_highlight {
            Some(range) => {
                let node = match source_file.syntax().covering_element(range) {
                    NodeOrToken::Node(it) => it,
                    NodeOrToken::Token(it) => it.parent(),
                };
                (node, range)
            }
            None => (source_file.syntax().clone(), source_file.syntax().text_range()),
        }
    };

    let mut bindings_shadow_count: FxHashMap<Name, u32> = FxHashMap::default();
    let mut res = Vec::new();

    let mut in_macro_call = None;

    for event in root.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(node) => {
                if node.text_range().intersection(&range_to_highlight).is_none() {
                    continue;
                }

                match node.kind() {
                    MACRO_CALL => {
                        in_macro_call = Some(node.clone());
                        if let Some(range) = highlight_macro(node) {
                            res.push(HighlightedRange {
                                range,
                                highlight: HighlightTag::Macro.into(),
                                binding_hash: None,
                            });
                        }
                    }
                    _ if in_macro_call.is_some() => {
                        if let Some(token) = node.as_token() {
                            if let Some((highlight, binding_hash)) = highlight_token_tree(
                                &sema,
                                &mut bindings_shadow_count,
                                token.clone(),
                            ) {
                                res.push(HighlightedRange {
                                    range: node.text_range(),
                                    highlight,
                                    binding_hash,
                                });
                            }
                        }
                    }
                    _ => {
                        if let Some((highlight, binding_hash)) =
                            highlight_node(&sema, &mut bindings_shadow_count, node.clone())
                        {
                            res.push(HighlightedRange {
                                range: node.text_range(),
                                highlight,
                                binding_hash,
                            });
                        }
                    }
                }
            }
            WalkEvent::Leave(node) => {
                if node.text_range().intersection(&range_to_highlight).is_none() {
                    continue;
                }

                if let Some(m) = in_macro_call.as_ref() {
                    if *m == node {
                        in_macro_call = None;
                    }
                }
            }
        }
    }

    res
}

fn highlight_macro(node: SyntaxElement) -> Option<TextRange> {
    let macro_call = ast::MacroCall::cast(node.as_node()?.clone())?;
    let path = macro_call.path()?;
    let name_ref = path.segment()?.name_ref()?;

    let range_start = name_ref.syntax().text_range().start();
    let mut range_end = name_ref.syntax().text_range().end();
    for sibling in path.syntax().siblings_with_tokens(Direction::Next) {
        match sibling.kind() {
            T![!] | IDENT => range_end = sibling.text_range().end(),
            _ => (),
        }
    }

    Some(TextRange::from_to(range_start, range_end))
}

fn highlight_token_tree(
    sema: &Semantics<RootDatabase>,
    bindings_shadow_count: &mut FxHashMap<Name, u32>,
    token: SyntaxToken,
) -> Option<(Highlight, Option<u64>)> {
    if token.parent().kind() != TOKEN_TREE {
        return None;
    }
    let token = sema.descend_into_macros(token.clone());
    let expanded = {
        let parent = token.parent();
        // We only care Name and Name_ref
        match (token.kind(), parent.kind()) {
            (IDENT, NAME) | (IDENT, NAME_REF) => parent.into(),
            _ => token.into(),
        }
    };

    highlight_node(sema, bindings_shadow_count, expanded)
}

fn highlight_node(
    sema: &Semantics<RootDatabase>,
    bindings_shadow_count: &mut FxHashMap<Name, u32>,
    node: SyntaxElement,
) -> Option<(Highlight, Option<u64>)> {
    let db = sema.db;
    let mut binding_hash = None;
    let highlight: Highlight = match node.kind() {
        FN_DEF => {
            bindings_shadow_count.clear();
            return None;
        }
        COMMENT => HighlightTag::Comment.into(),
        STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => HighlightTag::LiteralString.into(),
        ATTR => HighlightTag::Attribute.into(),
        // Special-case field init shorthand
        NAME_REF if node.parent().and_then(ast::RecordField::cast).is_some() => {
            HighlightTag::Field.into()
        }
        NAME_REF if node.ancestors().any(|it| it.kind() == ATTR) => return None,
        NAME_REF => {
            let name_ref = node.as_node().cloned().and_then(ast::NameRef::cast).unwrap();
            let name_kind = classify_name_ref(sema, &name_ref);
            match name_kind {
                Some(name_kind) => {
                    if let NameDefinition::Local(local) = &name_kind {
                        if let Some(name) = local.name(db) {
                            let shadow_count =
                                bindings_shadow_count.entry(name.clone()).or_default();
                            binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                        }
                    };

                    highlight_name(db, name_kind)
                }
                _ => return None,
            }
        }
        NAME => {
            let name = node.as_node().cloned().and_then(ast::Name::cast).unwrap();
            let name_kind = classify_name(sema, &name);

            if let Some(NameDefinition::Local(local)) = &name_kind {
                if let Some(name) = local.name(db) {
                    let shadow_count = bindings_shadow_count.entry(name.clone()).or_default();
                    *shadow_count += 1;
                    binding_hash = Some(calc_binding_hash(&name, *shadow_count))
                }
            };

            match name_kind {
                Some(name_kind) => highlight_name(db, name_kind),
                None => name.syntax().parent().map_or(HighlightTag::Function.into(), |x| {
                    match x.kind() {
                        STRUCT_DEF | ENUM_DEF | TRAIT_DEF | TYPE_ALIAS_DEF => {
                            HighlightTag::Type.into()
                        }
                        TYPE_PARAM => HighlightTag::TypeParam.into(),
                        RECORD_FIELD_DEF => HighlightTag::Field.into(),
                        _ => HighlightTag::Function.into(),
                    }
                }),
            }
        }
        INT_NUMBER | FLOAT_NUMBER => HighlightTag::LiteralNumeric.into(),
        BYTE => HighlightTag::LiteralByte.into(),
        CHAR => HighlightTag::LiteralChar.into(),
        LIFETIME => HighlightTag::TypeLifetime.into(),
        T![unsafe] => HighlightTag::Keyword | HighlightModifier::Unsafe,
        k if is_control_keyword(k) => HighlightTag::Keyword | HighlightModifier::Control,
        k if k.is_keyword() => HighlightTag::Keyword.into(),

        _ => return None,
    };

    return Some((highlight, binding_hash));

    fn calc_binding_hash(name: &Name, shadow_count: u32) -> u64 {
        fn hash<T: std::hash::Hash + std::fmt::Debug>(x: T) -> u64 {
            use std::{collections::hash_map::DefaultHasher, hash::Hasher};

            let mut hasher = DefaultHasher::new();
            x.hash(&mut hasher);
            hasher.finish()
        }

        hash((name, shadow_count))
    }
}

fn highlight_name(db: &RootDatabase, def: NameDefinition) -> Highlight {
    match def {
        NameDefinition::Macro(_) => HighlightTag::Macro,
        NameDefinition::StructField(_) => HighlightTag::Field,
        NameDefinition::ModuleDef(hir::ModuleDef::Module(_)) => HighlightTag::Module,
        NameDefinition::ModuleDef(hir::ModuleDef::Function(_)) => HighlightTag::Function,
        NameDefinition::ModuleDef(hir::ModuleDef::Adt(_)) => HighlightTag::Type,
        NameDefinition::ModuleDef(hir::ModuleDef::EnumVariant(_)) => HighlightTag::Constant,
        NameDefinition::ModuleDef(hir::ModuleDef::Const(_)) => HighlightTag::Constant,
        NameDefinition::ModuleDef(hir::ModuleDef::Static(_)) => HighlightTag::Constant,
        NameDefinition::ModuleDef(hir::ModuleDef::Trait(_)) => HighlightTag::Type,
        NameDefinition::ModuleDef(hir::ModuleDef::TypeAlias(_)) => HighlightTag::Type,
        NameDefinition::ModuleDef(hir::ModuleDef::BuiltinType(_)) => {
            return HighlightTag::Type | HighlightModifier::Builtin
        }
        NameDefinition::SelfType(_) => HighlightTag::TypeSelf,
        NameDefinition::TypeParam(_) => HighlightTag::TypeParam,
        NameDefinition::Local(local) => {
            let mut h = Highlight::new(HighlightTag::Variable);
            if local.is_mut(db) || local.ty(db).is_mutable_reference() {
                h |= HighlightModifier::Mutable;
            }
            return h;
        }
    }
    .into()
}

#[cfg(test)]
mod tests {
    use std::fs;

    use test_utils::{assert_eq_text, project_dir, read_text};

    use crate::{
        mock_analysis::{single_file, MockAnalysis},
        FileRange, TextRange,
    };

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
    foo::<i32>();
}

macro_rules! def_fn {
    ($($tt:tt)*) => {$($tt)*}
}

def_fn!{
    fn bar() -> u32 {
        100
    }
}

// comment
fn main() {
    println!("Hello, {}!", 92);

    let mut vec = Vec::new();
    if true {
        let x = 92;
        vec.push(Foo { x, y: 1 });
    }
    unsafe { vec.set_len(0); }

    let mut x = 42;
    let y = &mut x;
    let z = &y;

    y;
}

enum E<X> {
    V(X)
}

impl<X> E<X> {
    fn new<T>() -> E<T> {}
}
"#
            .trim(),
        );
        let dst_file = project_dir().join("crates/ra_ide/src/snapshots/highlighting.html");
        let actual_html = &analysis.highlight_as_html(file_id, false).unwrap();
        let expected_html = &read_text(&dst_file);
        fs::write(dst_file, &actual_html).unwrap();
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

fn bar() {
    let mut hello = "hello";
}
"#
            .trim(),
        );
        let dst_file = project_dir().join("crates/ra_ide/src/snapshots/rainbow_highlighting.html");
        let actual_html = &analysis.highlight_as_html(file_id, true).unwrap();
        let expected_html = &read_text(&dst_file);
        fs::write(dst_file, &actual_html).unwrap();
        assert_eq_text!(expected_html, actual_html);
    }

    #[test]
    fn accidentally_quadratic() {
        let file = project_dir().join("crates/ra_syntax/test_data/accidentally_quadratic");
        let src = fs::read_to_string(file).unwrap();

        let mut mock = MockAnalysis::new();
        let file_id = mock.add_file("/main.rs", &src);
        let host = mock.analysis_host();

        // let t = std::time::Instant::now();
        let _ = host.analysis().highlight(file_id).unwrap();
        // eprintln!("elapsed: {:?}", t.elapsed());
    }

    #[test]
    fn test_ranges() {
        let (analysis, file_id) = single_file(
            r#"
            #[derive(Clone, Debug)]
            struct Foo {
                pub x: i32,
                pub y: i32,
            }"#,
        );

        // The "x"
        let highlights = &analysis
            .highlight_range(FileRange {
                file_id,
                range: TextRange::offset_len(82.into(), 1.into()),
            })
            .unwrap();

        assert_eq!(&highlights[0].highlight.to_string(), "field");
    }
}
