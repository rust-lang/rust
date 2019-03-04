use insta::assert_debug_snapshot_matches;
use ra_ide_api::{
    mock_analysis::{single_file, single_file_with_position, single_file_with_range, MockAnalysis},
    AnalysisChange, CrateGraph, Edition::Edition2018, Query, NavigationTarget,
    ReferenceSearchResult,
};
use ra_syntax::SmolStr;

#[test]
fn test_unresolved_module_diagnostic() {
    let (analysis, file_id) = single_file("mod foo;");
    let diagnostics = analysis.diagnostics(file_id).unwrap();
    assert_debug_snapshot_matches!("unresolved_module_diagnostic", &diagnostics);
}

// FIXME: move this test to hir
#[test]
fn test_unresolved_module_diagnostic_no_diag_for_inline_mode() {
    let (analysis, file_id) = single_file("mod foo {}");
    let diagnostics = analysis.diagnostics(file_id).unwrap();
    assert!(diagnostics.is_empty());
}

#[test]
fn test_resolve_crate_root() {
    let mock = MockAnalysis::with_files(
        "
        //- /bar.rs
        mod foo;
        //- /foo.rs
        // empty <|>
    ",
    );
    let root_file = mock.id_of("/bar.rs");
    let mod_file = mock.id_of("/foo.rs");
    let mut host = mock.analysis_host();
    assert!(host.analysis().crate_for(mod_file).unwrap().is_empty());

    let mut crate_graph = CrateGraph::default();
    let crate_id = crate_graph.add_crate_root(root_file, Edition2018);
    let mut change = AnalysisChange::new();
    change.set_crate_graph(crate_graph);
    host.apply_change(change);

    assert_eq!(host.analysis().crate_for(mod_file).unwrap(), vec![crate_id]);
}

fn get_all_refs(text: &str) -> ReferenceSearchResult {
    let (analysis, position) = single_file_with_position(text);
    analysis.find_all_refs(position).unwrap().unwrap()
}

fn get_symbols_matching(text: &str, query: &str) -> Vec<NavigationTarget> {
    let (analysis, _) = single_file(text);
    analysis.symbol_search(Query::new(query.into())).unwrap()
}

#[test]
fn test_find_all_refs_for_local() {
    let code = r#"
    fn main() {
        let mut i = 1;
        let j = 1;
        i = i<|> + j;

        {
            i = 0;
        }

        i = 5;
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 5);
}

#[test]
fn test_find_all_refs_for_param_inside() {
    let code = r#"
    fn foo(i : u32) -> u32 {
        i<|>
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 2);
}

#[test]
fn test_find_all_refs_for_fn_param() {
    let code = r#"
    fn foo(i<|> : u32) -> u32 {
        i
    }"#;

    let refs = get_all_refs(code);
    assert_eq!(refs.len(), 2);
}

#[test]
fn test_world_symbols_with_no_container() {
    let code = r#"
    enum FooInner { }
    "#;

    let mut symbols = get_symbols_matching(code, "FooInner");

    let s = symbols.pop().unwrap();

    assert_eq!(s.name(), "FooInner");
    assert!(s.container_name().is_none());
}

#[test]
fn test_world_symbols_include_container_name() {
    let code = r#"
fn foo() {
    enum FooInner { }
}
    "#;

    let mut symbols = get_symbols_matching(code, "FooInner");

    let s = symbols.pop().unwrap();

    assert_eq!(s.name(), "FooInner");
    assert_eq!(s.container_name(), Some(&SmolStr::new("foo")));

    let code = r#"
mod foo {
    struct FooInner;
}
    "#;

    let mut symbols = get_symbols_matching(code, "FooInner");

    let s = symbols.pop().unwrap();

    assert_eq!(s.name(), "FooInner");
    assert_eq!(s.container_name(), Some(&SmolStr::new("foo")));
}

#[test]
fn test_syntax_tree_without_range() {
    // Basic syntax
    let (analysis, file_id) = single_file(r#"fn foo() {}"#);
    let syn = analysis.syntax_tree(file_id, None);

    assert_eq!(
        syn.trim(),
        r#"
SOURCE_FILE@[0; 11)
  FN_DEF@[0; 11)
    FN_KW@[0; 2)
    WHITESPACE@[2; 3)
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7)
      R_PAREN@[7; 8)
    WHITESPACE@[8; 9)
    BLOCK@[9; 11)
      L_CURLY@[9; 10)
      R_CURLY@[10; 11)
    "#
        .trim()
    );

    let (analysis, file_id) = single_file(
        r#"
fn test() {
    assert!("
    fn foo() {
    }
    ", "");
}"#
        .trim(),
    );
    let syn = analysis.syntax_tree(file_id, None);

    assert_eq!(
        syn.trim(),
        r#"
SOURCE_FILE@[0; 60)
  FN_DEF@[0; 60)
    FN_KW@[0; 2)
    WHITESPACE@[2; 3)
    NAME@[3; 7)
      IDENT@[3; 7) "test"
    PARAM_LIST@[7; 9)
      L_PAREN@[7; 8)
      R_PAREN@[8; 9)
    WHITESPACE@[9; 10)
    BLOCK@[10; 60)
      L_CURLY@[10; 11)
      WHITESPACE@[11; 16)
      EXPR_STMT@[16; 58)
        MACRO_CALL@[16; 57)
          PATH@[16; 22)
            PATH_SEGMENT@[16; 22)
              NAME_REF@[16; 22)
                IDENT@[16; 22) "assert"
          EXCL@[22; 23)
          TOKEN_TREE@[23; 57)
            L_PAREN@[23; 24)
            STRING@[24; 52)
            COMMA@[52; 53)
            WHITESPACE@[53; 54)
            STRING@[54; 56)
            R_PAREN@[56; 57)
        SEMI@[57; 58)
      WHITESPACE@[58; 59)
      R_CURLY@[59; 60)
    "#
        .trim()
    );
}

#[test]
fn test_syntax_tree_with_range() {
    let (analysis, range) = single_file_with_range(r#"<|>fn foo() {}<|>"#.trim());
    let syn = analysis.syntax_tree(range.file_id, Some(range.range));

    assert_eq!(
        syn.trim(),
        r#"
FN_DEF@[0; 11)
  FN_KW@[0; 2)
  WHITESPACE@[2; 3)
  NAME@[3; 6)
    IDENT@[3; 6) "foo"
  PARAM_LIST@[6; 8)
    L_PAREN@[6; 7)
    R_PAREN@[7; 8)
  WHITESPACE@[8; 9)
  BLOCK@[9; 11)
    L_CURLY@[9; 10)
    R_CURLY@[10; 11)
    "#
        .trim()
    );

    let (analysis, range) = single_file_with_range(
        r#"fn test() {
    <|>assert!("
    fn foo() {
    }
    ", "");<|>
}"#
        .trim(),
    );
    let syn = analysis.syntax_tree(range.file_id, Some(range.range));

    assert_eq!(
        syn.trim(),
        r#"
EXPR_STMT@[16; 58)
  MACRO_CALL@[16; 57)
    PATH@[16; 22)
      PATH_SEGMENT@[16; 22)
        NAME_REF@[16; 22)
          IDENT@[16; 22) "assert"
    EXCL@[22; 23)
    TOKEN_TREE@[23; 57)
      L_PAREN@[23; 24)
      STRING@[24; 52)
      COMMA@[52; 53)
      WHITESPACE@[53; 54)
      STRING@[54; 56)
      R_PAREN@[56; 57)
  SEMI@[57; 58)
    "#
        .trim()
    );
}

#[test]
fn test_syntax_tree_inside_string() {
    let (analysis, range) = single_file_with_range(
        r#"fn test() {
    assert!("
<|>fn foo() {
}<|>
fn bar() {
}
    ", "");
}"#
        .trim(),
    );
    let syn = analysis.syntax_tree(range.file_id, Some(range.range));
    assert_eq!(
        syn.trim(),
        r#"
SOURCE_FILE@[0; 12)
  FN_DEF@[0; 12)
    FN_KW@[0; 2)
    WHITESPACE@[2; 3)
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7)
      R_PAREN@[7; 8)
    WHITESPACE@[8; 9)
    BLOCK@[9; 12)
      L_CURLY@[9; 10)
      WHITESPACE@[10; 11)
      R_CURLY@[11; 12)
"#
        .trim()
    );

    // With a raw string
    let (analysis, range) = single_file_with_range(
        r###"fn test() {
    assert!(r#"
<|>fn foo() {
}<|>
fn bar() {
}
    "#, "");
}"###
            .trim(),
    );
    let syn = analysis.syntax_tree(range.file_id, Some(range.range));
    assert_eq!(
        syn.trim(),
        r#"
SOURCE_FILE@[0; 12)
  FN_DEF@[0; 12)
    FN_KW@[0; 2)
    WHITESPACE@[2; 3)
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7)
      R_PAREN@[7; 8)
    WHITESPACE@[8; 9)
    BLOCK@[9; 12)
      L_CURLY@[9; 10)
      WHITESPACE@[10; 11)
      R_CURLY@[11; 12)
"#
        .trim()
    );

    // With a raw string
    let (analysis, range) = single_file_with_range(
        r###"fn test() {
    assert!(r<|>#"
fn foo() {
}
fn bar() {
}"<|>#, "");
}"###
            .trim(),
    );
    let syn = analysis.syntax_tree(range.file_id, Some(range.range));
    assert_eq!(
        syn.trim(),
        r#"
SOURCE_FILE@[0; 25)
  FN_DEF@[0; 12)
    FN_KW@[0; 2)
    WHITESPACE@[2; 3)
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7)
      R_PAREN@[7; 8)
    WHITESPACE@[8; 9)
    BLOCK@[9; 12)
      L_CURLY@[9; 10)
      WHITESPACE@[10; 11)
      R_CURLY@[11; 12)
  WHITESPACE@[12; 13)
  FN_DEF@[13; 25)
    FN_KW@[13; 15)
    WHITESPACE@[15; 16)
    NAME@[16; 19)
      IDENT@[16; 19) "bar"
    PARAM_LIST@[19; 21)
      L_PAREN@[19; 20)
      R_PAREN@[20; 21)
    WHITESPACE@[21; 22)
    BLOCK@[22; 25)
      L_CURLY@[22; 23)
      WHITESPACE@[23; 24)
      R_CURLY@[24; 25)

"#
        .trim()
    );
}
