use std::collections::HashMap;

use syntax::{ast, AstNode};
use test_utils::extract_annotations;
use tt::{
    buffer::{TokenBuffer, TokenTreeRef},
    Leaf, Punct, Spacing, SpanAnchor, SyntaxContext,
};

use super::syntax_node_to_token_tree;

fn check_punct_spacing(fixture: &str) {
    type SpanData = tt::SpanData<DummyFile, DummyCtx>;

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    struct DummyFile;
    impl SpanAnchor for DummyFile {
        const DUMMY: Self = DummyFile;
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    struct DummyCtx;
    impl SyntaxContext for DummyCtx {
        const DUMMY: Self = DummyCtx;
    }

    let source_file = ast::SourceFile::parse(fixture).ok().unwrap();
    let subtree =
        syntax_node_to_token_tree(source_file.syntax(), DummyFile, 0.into(), &Default::default());
    let mut annotations: HashMap<_, _> = extract_annotations(fixture)
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

    let buf = TokenBuffer::from_subtree(&subtree);
    let mut cursor = buf.begin();
    while !cursor.eof() {
        while let Some(token_tree) = cursor.token_tree() {
            if let TokenTreeRef::Leaf(
                Leaf::Punct(Punct { spacing, span: SpanData { range, .. }, .. }),
                _,
            ) = token_tree
            {
                if let Some(expected) = annotations.remove(range) {
                    assert_eq!(expected, *spacing);
                }
            }
            cursor = cursor.bump_subtree();
        }
        cursor = cursor.bump();
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
