//! FIXME: write short doc here

use hir::{Function, HirDisplay, SourceAnalyzer, SourceBinder};
use once_cell::unsync::Lazy;
use ra_ide_db::RootDatabase;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, ArgListOwner, AstNode, TypeAscriptionOwner},
    match_ast, SmolStr, SourceFile, SyntaxNode, TextRange,
};

use crate::{FileId, FunctionSignature};

#[derive(Debug, PartialEq, Eq)]
pub enum InlayKind {
    TypeHint,
    ParameterHint,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub kind: InlayKind,
    pub label: SmolStr,
}

pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    file: &SourceFile,
    max_inlay_hint_length: Option<usize>,
) -> Vec<InlayHint> {
    let mut sb = SourceBinder::new(db);
    let mut res = Vec::new();
    for node in file.syntax().descendants() {
        get_inlay_hints(&mut res, &mut sb, file_id, &node, max_inlay_hint_length);
    }
    res
}

fn get_inlay_hints(
    acc: &mut Vec<InlayHint>,
    sb: &mut SourceBinder<RootDatabase>,
    file_id: FileId,
    node: &SyntaxNode,
    max_inlay_hint_length: Option<usize>,
) -> Option<()> {
    let _p = profile("get_inlay_hints");
    let db = sb.db;
    let analyzer = Lazy::new(move || sb.analyze(hir::InFile::new(file_id.into(), node), None));
    match_ast! {
        match node {
            ast::CallExpr(it) => {
                get_param_name_hints(acc, db, &analyzer, ast::Expr::from(it));
            },
            ast::MethodCallExpr(it) => {
                get_param_name_hints(acc, db, &analyzer, ast::Expr::from(it));
            },
            ast::BindPat(it) => {
                if should_not_display_type_hint(&it) {
                    return None;
                }
                let pat = ast::Pat::from(it);
                let ty = analyzer.type_of_pat(db, &pat)?;
                if ty.is_unknown() {
                    return None;
                }

                acc.push(
                    InlayHint {
                        range: pat.syntax().text_range(),
                        kind: InlayKind::TypeHint,
                        label: ty.display_truncated(db, max_inlay_hint_length).to_string().into(),
                    }
                );
            },
            _ => (),
        }
    };
    Some(())
}

fn should_not_display_type_hint(bind_pat: &ast::BindPat) -> bool {
    for node in bind_pat.syntax().ancestors() {
        match_ast! {
            match node {
                ast::LetStmt(it) => {
                    return it.ascribed_type().is_some()
                },
                ast::Param(it) => {
                    return it.ascribed_type().is_some()
                },
                _ => (),
            }
        }
    }
    false
}

fn get_param_name_hints(
    acc: &mut Vec<InlayHint>,
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    expr: ast::Expr,
) -> Option<()> {
    let args = match &expr {
        ast::Expr::CallExpr(expr) => expr.arg_list()?.args(),
        ast::Expr::MethodCallExpr(expr) => expr.arg_list()?.args(),
        _ => return None,
    }
    .into_iter()
    // we need args len to determine whether to skip or not the &self parameter
    .collect::<Vec<_>>();

    let (has_self_param, fn_signature) = get_fn_signature(db, analyzer, &expr)?;
    let parameters = if has_self_param && fn_signature.parameter_names.len() > args.len() {
        fn_signature.parameter_names.into_iter().skip(1)
    } else {
        fn_signature.parameter_names.into_iter().skip(0)
    };

    let hints =
        parameters
            .zip(args)
            .filter_map(|(param, arg)| {
                if !param.is_empty() {
                    Some((arg.syntax().text_range(), param))
                } else {
                    None
                }
            })
            .map(|(range, param_name)| InlayHint {
                range,
                kind: InlayKind::ParameterHint,
                label: param_name.into(),
            });

    acc.extend(hints);
    Some(())
}

fn get_fn_signature(
    db: &RootDatabase,
    analyzer: &SourceAnalyzer,
    expr: &ast::Expr,
) -> Option<(bool, FunctionSignature)> {
    match expr {
        ast::Expr::CallExpr(expr) => {
            // FIXME: Type::as_callable is broken for closures
            let callable_def = analyzer.type_of(db, &expr.expr()?)?.as_callable()?;
            match callable_def {
                hir::CallableDef::FunctionId(it) => {
                    let fn_def: Function = it.into();
                    Some((fn_def.has_self_param(db), FunctionSignature::from_hir(db, fn_def)))
                }
                hir::CallableDef::StructId(it) => FunctionSignature::from_struct(db, it.into())
                    .map(|signature| (false, signature)),
                hir::CallableDef::EnumVariantId(it) => {
                    FunctionSignature::from_enum_variant(db, it.into())
                        .map(|signature| (false, signature))
                }
            }
        }
        ast::Expr::MethodCallExpr(expr) => {
            let fn_def = analyzer.resolve_method_call(&expr)?;
            Some((fn_def.has_self_param(db), FunctionSignature::from_hir(db, fn_def)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;

    use crate::mock_analysis::single_file;

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        let (analysis, file_id) = single_file(
            r#"
struct Test<K, T = u8> {
    k: K,
    t: T,
}

fn main() {
    let zz = Test { t: 23, k: 33 };
    let zz_ref = &zz;
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [69; 71),
                kind: TypeHint,
                label: "Test<i32>",
            },
            InlayHint {
                range: [105; 111),
                kind: TypeHint,
                label: "&Test<i32>",
            },
        ]
        "###
        );
    }

    #[test]
    fn let_statement() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    struct InnerStruct {}

    let test = 54;
    let test: i32 = 33;
    let mut test = 33;
    let _ = 22;
    let test = "test";
    let test = InnerStruct {};

    let test = vec![222];
    let test: Vec<_> = (0..3).collect();
    let test = (0..3).collect::<Vec<i128>>();
    let test = (0..3).collect::<Vec<_>>();

    let mut test = Vec::new();
    test.push(333);

    let test = (42, 'a');
    let (a, (b, c, (d, e), f)) = (2, (3, 4, (6.6, 7.7), 5));
    let &x = &92;
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [193; 197),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [236; 244),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [275; 279),
                kind: TypeHint,
                label: "&str",
            },
            InlayHint {
                range: [539; 543),
                kind: TypeHint,
                label: "(i32, char)",
            },
            InlayHint {
                range: [566; 567),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [570; 571),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [573; 574),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [577; 578),
                kind: TypeHint,
                label: "f64",
            },
            InlayHint {
                range: [580; 581),
                kind: TypeHint,
                label: "f64",
            },
            InlayHint {
                range: [584; 585),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [627; 628),
                kind: TypeHint,
                label: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn closure_parameters() {
        let (analysis, file_id) = single_file(
            r#"
fn main() {
    let mut start = 0;
    (0..2).for_each(|increment| {
        start += increment;
    });

    let multiply = |a, b, c, d| a * b * c * d;
    let _: i32 = multiply(1, 2, 3, 4);
    let multiply_ref = &multiply;

    let return_42 = || 42;
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [21; 30),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [57; 66),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [115; 123),
                kind: TypeHint,
                label: "|…| -> i32",
            },
            InlayHint {
                range: [127; 128),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [130; 131),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [133; 134),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [136; 137),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [201; 213),
                kind: TypeHint,
                label: "&|…| -> i32",
            },
            InlayHint {
                range: [236; 245),
                kind: TypeHint,
                label: "|| -> i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn for_expression() {
        let (analysis, file_id) = single_file(
            r#"
fn main() {
    let mut start = 0;
    for increment in 0..2 {
        start += increment;
    }
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [21; 30),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [44; 53),
                kind: TypeHint,
                label: "i32",
            },
        ]
        "###
        );
    }

    #[test]
    fn if_expr() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    let test = CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 });
    if let CustomOption::None = &test {};
    if let test = &test {};
    if let CustomOption::Some(test) = &test {};
    if let CustomOption::Some(Test { a, b }) = &test {};
    if let CustomOption::Some(Test { a: x, b: y }) = &test {};
    if let CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) = &test {};
    if let CustomOption::Some(Test { a: CustomOption::None, b: y }) = &test {};
    if let CustomOption::Some(Test { b: y, .. }) = &test {};

    if test == CustomOption::None {}
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [166; 170),
                kind: TypeHint,
                label: "CustomOption<Test>",
            },
            InlayHint {
                range: [287; 291),
                kind: TypeHint,
                label: "&CustomOption<Test>",
            },
            InlayHint {
                range: [334; 338),
                kind: TypeHint,
                label: "&Test",
            },
            InlayHint {
                range: [389; 390),
                kind: TypeHint,
                label: "&CustomOption<u32>",
            },
            InlayHint {
                range: [392; 393),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [449; 450),
                kind: TypeHint,
                label: "&CustomOption<u32>",
            },
            InlayHint {
                range: [455; 456),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [531; 532),
                kind: TypeHint,
                label: "&u32",
            },
            InlayHint {
                range: [538; 539),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [618; 619),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [675; 676),
                kind: TypeHint,
                label: "&u8",
            },
        ]
        "###
        );
    }

    #[test]
    fn while_expr() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    let test = CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 });
    while let CustomOption::None = &test {};
    while let test = &test {};
    while let CustomOption::Some(test) = &test {};
    while let CustomOption::Some(Test { a, b }) = &test {};
    while let CustomOption::Some(Test { a: x, b: y }) = &test {};
    while let CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) = &test {};
    while let CustomOption::Some(Test { a: CustomOption::None, b: y }) = &test {};
    while let CustomOption::Some(Test { b: y, .. }) = &test {};

    while test == CustomOption::None {}
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [166; 170),
                kind: TypeHint,
                label: "CustomOption<Test>",
            },
            InlayHint {
                range: [293; 297),
                kind: TypeHint,
                label: "&CustomOption<Test>",
            },
            InlayHint {
                range: [343; 347),
                kind: TypeHint,
                label: "&Test",
            },
            InlayHint {
                range: [401; 402),
                kind: TypeHint,
                label: "&CustomOption<u32>",
            },
            InlayHint {
                range: [404; 405),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [464; 465),
                kind: TypeHint,
                label: "&CustomOption<u32>",
            },
            InlayHint {
                range: [470; 471),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [549; 550),
                kind: TypeHint,
                label: "&u32",
            },
            InlayHint {
                range: [556; 557),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [639; 640),
                kind: TypeHint,
                label: "&u8",
            },
            InlayHint {
                range: [699; 700),
                kind: TypeHint,
                label: "&u8",
            },
        ]
        "###
        );
    }

    #[test]
    fn match_arm_list() {
        let (analysis, file_id) = single_file(
            r#"
#[derive(PartialEq)]
enum CustomOption<T> {
    None,
    Some(T),
}

#[derive(PartialEq)]
struct Test {
    a: CustomOption<u32>,
    b: u8,
}

fn main() {
    match CustomOption::Some(Test { a: CustomOption::Some(3), b: 1 }) {
        CustomOption::None => (),
        test => (),
        CustomOption::Some(test) => (),
        CustomOption::Some(Test { a, b }) => (),
        CustomOption::Some(Test { a: x, b: y }) => (),
        CustomOption::Some(Test { a: CustomOption::Some(x), b: y }) => (),
        CustomOption::Some(Test { a: CustomOption::None, b: y }) => (),
        CustomOption::Some(Test { b: y, .. }) => (),
        _ => {}
    }
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [272; 276),
                kind: TypeHint,
                label: "CustomOption<Test>",
            },
            InlayHint {
                range: [311; 315),
                kind: TypeHint,
                label: "Test",
            },
            InlayHint {
                range: [358; 359),
                kind: TypeHint,
                label: "CustomOption<u32>",
            },
            InlayHint {
                range: [361; 362),
                kind: TypeHint,
                label: "u8",
            },
            InlayHint {
                range: [410; 411),
                kind: TypeHint,
                label: "CustomOption<u32>",
            },
            InlayHint {
                range: [416; 417),
                kind: TypeHint,
                label: "u8",
            },
            InlayHint {
                range: [484; 485),
                kind: TypeHint,
                label: "u32",
            },
            InlayHint {
                range: [491; 492),
                kind: TypeHint,
                label: "u8",
            },
            InlayHint {
                range: [563; 564),
                kind: TypeHint,
                label: "u8",
            },
            InlayHint {
                range: [612; 613),
                kind: TypeHint,
                label: "u8",
            },
        ]
        "###
        );
    }

    #[test]
    fn hint_truncation() {
        let (analysis, file_id) = single_file(
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
    let b = VeryLongOuterName(0usize);
    let c = Smol(Smol(0u32))
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, Some(8)).unwrap(), @r###"
        [
            InlayHint {
                range: [74; 75),
                kind: TypeHint,
                label: "Smol<u32>",
            },
            InlayHint {
                range: [98; 99),
                kind: TypeHint,
                label: "VeryLongOuterName<…>",
            },
            InlayHint {
                range: [137; 138),
                kind: TypeHint,
                label: "Smol<Smol<…>>",
            },
        ]
        "###
        );
    }

    #[test]
    fn function_call_parameter_hint() {
        let (analysis, file_id) = single_file(
            r#"
struct Test {}

impl Test {
    fn method(&self, mut param: i32) -> i32 {
        param * 2
    }
}

fn test_func(mut foo: i32, bar: i32, msg: &str, _: i32, last: i32) -> i32 {
    foo + bar
}

fn main() {
    let not_literal = 1;
    let _: i32 = test_func(1, 2, "hello", 3, not_literal);
    let t: Test = Test {};
    t.method(123);
    Test::method(&t, 3456);
}"#,
        );

        assert_debug_snapshot!(analysis.inlay_hints(file_id, None).unwrap(), @r###"
        [
            InlayHint {
                range: [215; 226),
                kind: TypeHint,
                label: "i32",
            },
            InlayHint {
                range: [259; 260),
                kind: ParameterHint,
                label: "foo",
            },
            InlayHint {
                range: [262; 263),
                kind: ParameterHint,
                label: "bar",
            },
            InlayHint {
                range: [265; 272),
                kind: ParameterHint,
                label: "msg",
            },
            InlayHint {
                range: [277; 288),
                kind: ParameterHint,
                label: "last",
            },
            InlayHint {
                range: [331; 334),
                kind: ParameterHint,
                label: "param",
            },
            InlayHint {
                range: [354; 356),
                kind: ParameterHint,
                label: "&self",
            },
            InlayHint {
                range: [358; 362),
                kind: ParameterHint,
                label: "param",
            },
        ]
        "###
        );
    }
}
