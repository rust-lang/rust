use expect_test::expect;
use rustc_hash::FxHashMap;
use span::Span;
use syntax::{AstNode, ast};
use test_utils::extract_annotations;
use tt::{Leaf, Punct, Spacing, buffer::Cursor};

use crate::{
    DocCommentDesugarMode,
    dummy_test_span_utils::{DUMMY, DummyTestSpanMap},
    parse_to_token_tree_static_span, syntax_node_to_token_tree, token_tree_to_syntax_node,
};

fn check_punct_spacing(fixture: &str) {
    let source_file = ast::SourceFile::parse(fixture, span::Edition::CURRENT).ok().unwrap();
    let subtree = syntax_node_to_token_tree(
        source_file.syntax(),
        DummyTestSpanMap,
        DUMMY,
        DocCommentDesugarMode::Mbe,
    );
    let mut annotations: FxHashMap<_, _> = extract_annotations(fixture)
        .into_iter()
        .map(|(range, annotation)| {
            let spacing = match annotation.as_str() {
                "Alone" => Spacing::Alone,
                "Joint" => Spacing::Joint,
                a => panic!("unknown annotation: {a}"),
            };
            (range, spacing)
        })
        .collect();

    let mut cursor = Cursor::new(subtree.as_token_trees());
    while !cursor.eof() {
        while let Some(token_tree) = cursor.token_tree() {
            if let tt::TokenTree::Leaf(Leaf::Punct(Punct {
                spacing, span: Span { range, .. }, ..
            })) = token_tree
                && let Some(expected) = annotations.remove(&range)
            {
                assert_eq!(expected, spacing);
            }
            cursor.bump();
        }
        cursor.bump_or_end();
    }

    assert!(annotations.is_empty(), "unchecked annotations: {annotations:?}");
}

#[test]
fn punct_spacing() {
    check_punct_spacing(
        r#"
fn main() {
    0+0;
   //^ Alone
    0+(0);
   //^ Alone
    0<=0;
   //^ Joint
   // ^ Alone
    0<=(0);
   // ^ Alone
    a=0;
   //^ Alone
    a=(0);
   //^ Alone
    a+=0;
   //^ Joint
   // ^ Alone
    a+=(0);
   // ^ Alone
    a&&b;
   //^ Joint
   // ^ Alone
    a&&(b);
   // ^ Alone
    foo::bar;
   //  ^ Joint
   //   ^ Alone
    use foo::{bar,baz,};
   //       ^ Alone
   //            ^ Alone
   //                ^ Alone
    struct Struct<'a> {};
   //            ^ Joint
   //             ^ Joint
    Struct::<0>;
   //       ^ Alone
    Struct::<{0}>;
   //       ^ Alone
    ;;
  //^ Joint
  // ^ Alone
}
        "#,
    );
}

#[test]
fn scientific_notation_field_access_recovers() {
    let tt = parse_to_token_tree_static_span(
        span::Edition::CURRENT,
        DUMMY,
        r#"
fn main() {
    s.00E+10;
}
"#,
    )
    .unwrap();

    let (parse, _) = token_tree_to_syntax_node(&tt, parser::TopEntryPoint::SourceFile, &mut |_| {
        span::Edition::CURRENT
    });
    let parse = parse.cast::<ast::SourceFile>().unwrap();

    expect![[r#"
        SOURCE_FILE@0..19
          FN@0..19
            FN_KW@0..2 "fn"
            NAME@2..6
              IDENT@2..6 "main"
            PARAM_LIST@6..8
              L_PAREN@6..7 "("
              R_PAREN@7..8 ")"
            BLOCK_EXPR@8..19
              STMT_LIST@8..19
                L_CURLY@8..9 "{"
                EXPR_STMT@9..18
                  FIELD_EXPR@9..17
                    FIELD_EXPR@9..17
                      PATH_EXPR@9..10
                        PATH@9..10
                          PATH_SEGMENT@9..10
                            NAME_REF@9..10
                              IDENT@9..10 "s"
                      DOT@10..11 "."
                      ERROR@11..17
                        FLOAT_NUMBER@11..17 "00E+10"
                  SEMICOLON@17..18 ";"
                R_CURLY@18..19 "}"
        error 11..11: illegal float literal
    "#]]
    .assert_eq(&parse.debug_dump());
}
