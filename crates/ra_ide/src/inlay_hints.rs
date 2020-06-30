use hir::{Adt, HirDisplay, Semantics, Type};
use ra_ide_db::RootDatabase;
use ra_prof::profile;
use ra_syntax::{
    ast::{self, ArgListOwner, AstNode, TypeAscriptionOwner},
    match_ast, Direction, NodeOrToken, SmolStr, SyntaxKind, TextRange, T,
};

use crate::{FileId, FunctionSignature};
use stdx::to_lower_snake_case;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InlayHintsConfig {
    pub type_hints: bool,
    pub parameter_hints: bool,
    pub chaining_hints: bool,
    pub max_length: Option<usize>,
}

impl Default for InlayHintsConfig {
    fn default() -> Self {
        Self { type_hints: true, parameter_hints: true, chaining_hints: true, max_length: None }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlayKind {
    TypeHint,
    ParameterHint,
    ChainingHint,
}

#[derive(Debug)]
pub struct InlayHint {
    pub range: TextRange,
    pub kind: InlayKind,
    pub label: SmolStr,
}

// Feature: Inlay Hints
//
// rust-analyzer shows additional information inline with the source code.
// Editors usually render this using read-only virtual text snippets interspersed with code.
//
// rust-analyzer shows hits for
//
// * types of local variables
// * names of function arguments
// * types of chained expressions
//
// **Note:** VS Code does not have native support for inlay hints https://github.com/microsoft/vscode/issues/16221[yet] and the hints are implemented using decorations.
// This approach has limitations, the caret movement and bracket highlighting near the edges of the hint may be weird:
// https://github.com/rust-analyzer/rust-analyzer/issues/1623[1], https://github.com/rust-analyzer/rust-analyzer/issues/3453[2].
//
// |===
// | Editor  | Action Name
//
// | VS Code | **Rust Analyzer: Toggle inlay hints*
// |===
pub(crate) fn inlay_hints(
    db: &RootDatabase,
    file_id: FileId,
    config: &InlayHintsConfig,
) -> Vec<InlayHint> {
    let _p = profile("inlay_hints");
    let sema = Semantics::new(db);
    let file = sema.parse(file_id);

    let mut res = Vec::new();
    for node in file.syntax().descendants() {
        if let Some(expr) = ast::Expr::cast(node.clone()) {
            get_chaining_hints(&mut res, &sema, config, expr);
        }

        match_ast! {
            match node {
                ast::CallExpr(it) => { get_param_name_hints(&mut res, &sema, config, ast::Expr::from(it)); },
                ast::MethodCallExpr(it) => { get_param_name_hints(&mut res, &sema, config, ast::Expr::from(it)); },
                ast::BindPat(it) => { get_bind_pat_hints(&mut res, &sema, config, it); },
                _ => (),
            }
        }
    }
    res
}

fn get_chaining_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    expr: ast::Expr,
) -> Option<()> {
    if !config.chaining_hints {
        return None;
    }

    if matches!(expr, ast::Expr::RecordLit(_)) {
        return None;
    }

    let mut tokens = expr
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter_map(NodeOrToken::into_token)
        .filter(|t| match t.kind() {
            SyntaxKind::WHITESPACE if !t.text().contains('\n') => false,
            SyntaxKind::COMMENT => false,
            _ => true,
        });

    // Chaining can be defined as an expression whose next sibling tokens are newline and dot
    // Ignoring extra whitespace and comments
    let next = tokens.next()?.kind();
    let next_next = tokens.next()?.kind();
    if next == SyntaxKind::WHITESPACE && next_next == T![.] {
        let ty = sema.type_of_expr(&expr)?;
        if ty.is_unknown() {
            return None;
        }
        if matches!(expr, ast::Expr::PathExpr(_)) {
            if let Some(Adt::Struct(st)) = ty.as_adt() {
                if st.fields(sema.db).is_empty() {
                    return None;
                }
            }
        }
        let label = ty.display_truncated(sema.db, config.max_length).to_string();
        acc.push(InlayHint {
            range: expr.syntax().text_range(),
            kind: InlayKind::ChainingHint,
            label: label.into(),
        });
    }
    Some(())
}

fn get_param_name_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    expr: ast::Expr,
) -> Option<()> {
    if !config.parameter_hints {
        return None;
    }

    let args = match &expr {
        ast::Expr::CallExpr(expr) => expr.arg_list()?.args(),
        ast::Expr::MethodCallExpr(expr) => expr.arg_list()?.args(),
        _ => return None,
    };

    let fn_signature = get_fn_signature(sema, &expr)?;
    let n_params_to_skip =
        if fn_signature.has_self_param && matches!(&expr, ast::Expr::MethodCallExpr(_)) {
            1
        } else {
            0
        };
    let hints = fn_signature
        .parameter_names
        .iter()
        .skip(n_params_to_skip)
        .zip(args)
        .filter(|(param, arg)| should_show_param_name_hint(sema, &fn_signature, param, &arg))
        .map(|(param_name, arg)| InlayHint {
            range: arg.syntax().text_range(),
            kind: InlayKind::ParameterHint,
            label: param_name.into(),
        });

    acc.extend(hints);
    Some(())
}

fn get_bind_pat_hints(
    acc: &mut Vec<InlayHint>,
    sema: &Semantics<RootDatabase>,
    config: &InlayHintsConfig,
    pat: ast::BindPat,
) -> Option<()> {
    if !config.type_hints {
        return None;
    }

    let ty = sema.type_of_pat(&pat.clone().into())?;

    if should_not_display_type_hint(sema.db, &pat, &ty) {
        return None;
    }

    acc.push(InlayHint {
        range: pat.syntax().text_range(),
        kind: InlayKind::TypeHint,
        label: ty.display_truncated(sema.db, config.max_length).to_string().into(),
    });
    Some(())
}

fn pat_is_enum_variant(db: &RootDatabase, bind_pat: &ast::BindPat, pat_ty: &Type) -> bool {
    if let Some(Adt::Enum(enum_data)) = pat_ty.as_adt() {
        let pat_text = bind_pat.to_string();
        enum_data
            .variants(db)
            .into_iter()
            .map(|variant| variant.name(db).to_string())
            .any(|enum_name| enum_name == pat_text)
    } else {
        false
    }
}

fn should_not_display_type_hint(db: &RootDatabase, bind_pat: &ast::BindPat, pat_ty: &Type) -> bool {
    if pat_ty.is_unknown() {
        return true;
    }

    if let Some(Adt::Struct(s)) = pat_ty.as_adt() {
        if s.fields(db).is_empty() && s.name(db).to_string() == bind_pat.to_string() {
            return true;
        }
    }

    for node in bind_pat.syntax().ancestors() {
        match_ast! {
            match node {
                ast::LetStmt(it) => {
                    return it.ascribed_type().is_some()
                },
                ast::Param(it) => {
                    return it.ascribed_type().is_some()
                },
                ast::MatchArm(_it) => {
                    return pat_is_enum_variant(db, bind_pat, pat_ty);
                },
                ast::IfExpr(it) => {
                    return it.condition().and_then(|condition| condition.pat()).is_some()
                        && pat_is_enum_variant(db, bind_pat, pat_ty);
                },
                ast::WhileExpr(it) => {
                    return it.condition().and_then(|condition| condition.pat()).is_some()
                        && pat_is_enum_variant(db, bind_pat, pat_ty);
                },
                _ => (),
            }
        }
    }
    false
}

fn should_show_param_name_hint(
    sema: &Semantics<RootDatabase>,
    fn_signature: &FunctionSignature,
    param_name: &str,
    argument: &ast::Expr,
) -> bool {
    let param_name = param_name.trim_start_matches('_');
    if param_name.is_empty()
        || Some(param_name) == fn_signature.name.as_ref().map(|s| s.trim_start_matches('_'))
        || is_argument_similar_to_param_name(sema, argument, param_name)
        || param_name.starts_with("ra_fixture")
    {
        return false;
    }

    let parameters_len = if fn_signature.has_self_param {
        fn_signature.parameters.len() - 1
    } else {
        fn_signature.parameters.len()
    };

    // avoid displaying hints for common functions like map, filter, etc.
    // or other obvious words used in std
    !(parameters_len == 1 && is_obvious_param(param_name))
}

fn is_argument_similar_to_param_name(
    sema: &Semantics<RootDatabase>,
    argument: &ast::Expr,
    param_name: &str,
) -> bool {
    if is_enum_name_similar_to_param_name(sema, argument, param_name) {
        return true;
    }
    match get_string_representation(argument) {
        None => false,
        Some(repr) => {
            let argument_string = repr.trim_start_matches('_');
            argument_string.starts_with(param_name) || argument_string.ends_with(param_name)
        }
    }
}

fn is_enum_name_similar_to_param_name(
    sema: &Semantics<RootDatabase>,
    argument: &ast::Expr,
    param_name: &str,
) -> bool {
    match sema.type_of_expr(argument).and_then(|t| t.as_adt()) {
        Some(Adt::Enum(e)) => to_lower_snake_case(&e.name(sema.db).to_string()) == param_name,
        _ => false,
    }
}

fn get_string_representation(expr: &ast::Expr) -> Option<String> {
    match expr {
        ast::Expr::MethodCallExpr(method_call_expr) => {
            Some(method_call_expr.name_ref()?.to_string())
        }
        ast::Expr::RefExpr(ref_expr) => get_string_representation(&ref_expr.expr()?),
        _ => Some(expr.to_string()),
    }
}

fn is_obvious_param(param_name: &str) -> bool {
    let is_obvious_param_name =
        matches!(param_name, "predicate" | "value" | "pat" | "rhs" | "other");
    param_name.len() == 1 || is_obvious_param_name
}

fn get_fn_signature(sema: &Semantics<RootDatabase>, expr: &ast::Expr) -> Option<FunctionSignature> {
    match expr {
        ast::Expr::CallExpr(expr) => {
            // FIXME: Type::as_callable is broken for closures
            let callable_def = sema.type_of_expr(&expr.expr()?)?.as_callable()?;
            match callable_def {
                hir::CallableDef::FunctionId(it) => {
                    Some(FunctionSignature::from_hir(sema.db, it.into()))
                }
                hir::CallableDef::StructId(it) => {
                    FunctionSignature::from_struct(sema.db, it.into())
                }
                hir::CallableDef::EnumVariantId(it) => {
                    FunctionSignature::from_enum_variant(sema.db, it.into())
                }
            }
        }
        ast::Expr::MethodCallExpr(expr) => {
            let fn_def = sema.resolve_method_call(&expr)?;
            Some(FunctionSignature::from_hir(sema.db, fn_def))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};
    use test_utils::extract_annotations;

    use crate::{inlay_hints::InlayHintsConfig, mock_analysis::single_file};

    fn check(ra_fixture: &str) {
        check_with_config(ra_fixture, InlayHintsConfig::default());
    }

    fn check_with_config(ra_fixture: &str, config: InlayHintsConfig) {
        let (analysis, file_id) = single_file(ra_fixture);
        let expected = extract_annotations(&*analysis.file_text(file_id).unwrap());
        let inlay_hints = analysis.inlay_hints(file_id, &config).unwrap();
        let actual =
            inlay_hints.into_iter().map(|it| (it.range, it.label.to_string())).collect::<Vec<_>>();
        assert_eq!(expected, actual);
    }

    fn check_expect(ra_fixture: &str, config: InlayHintsConfig, expect: Expect) {
        let (analysis, file_id) = single_file(ra_fixture);
        let inlay_hints = analysis.inlay_hints(file_id, &config).unwrap();
        expect.assert_debug_eq(&inlay_hints)
    }

    #[test]
    fn param_hints_only() {
        check_with_config(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(
        4,
      //^ a
        4,
      //^ b
    );
}"#,
            InlayHintsConfig {
                parameter_hints: true,
                type_hints: false,
                chaining_hints: false,
                max_length: None,
            },
        );
    }

    #[test]
    fn hints_disabled() {
        check_with_config(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
}"#,
            InlayHintsConfig {
                type_hints: false,
                parameter_hints: false,
                chaining_hints: false,
                max_length: None,
            },
        );
    }

    #[test]
    fn type_hints_only() {
        check_with_config(
            r#"
fn foo(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _x = foo(4, 4);
      //^^ i32
}"#,
            InlayHintsConfig {
                type_hints: true,
                parameter_hints: false,
                chaining_hints: false,
                max_length: None,
            },
        );
    }

    #[test]
    fn default_generic_types_should_not_be_displayed() {
        check(
            r#"
struct Test<K, T = u8> { k: K, t: T }

fn main() {
    let zz = Test { t: 23u8, k: 33 };
      //^^ Test<i32>
    let zz_ref = &zz;
      //^^^^^^ &Test<i32>
}"#,
        );
    }

    #[test]
    fn let_statement() {
        check(
            r#"
#[derive(PartialEq)]
enum Option<T> { None, Some(T) }

#[derive(PartialEq)]
struct Test { a: Option<u32>, b: u8 }

fn main() {
    struct InnerStruct {}

    let test = 54;
      //^^^^ i32
    let test: i32 = 33;
    let mut test = 33;
      //^^^^^^^^ i32
    let _ = 22;
    let test = "test";
      //^^^^ &str
    let test = InnerStruct {};

    let test = unresolved();

    let test = (42, 'a');
      //^^^^ (i32, char)
    let (a,    (b,     (c,)) = (2, (3, (9.2,));
       //^ i32  ^ i32   ^ f64
    let &x = &92;
       //^ i32
}"#,
        );
    }

    #[test]
    fn closure_parameters() {
        check(
            r#"
fn main() {
    let mut start = 0;
      //^^^^^^^^^ i32
    (0..2).for_each(|increment| { start += increment; });
                   //^^^^^^^^^ i32

    let multiply =
      //^^^^^^^^ |…| -> i32
      | a,     b| a * b
      //^ i32  ^ i32
    ;

    let _: i32 = multiply(1, 2);
    let multiply_ref = &multiply;
      //^^^^^^^^^^^^ &|…| -> i32

    let return_42 = || 42;
      //^^^^^^^^^ || -> i32
}"#,
        );
    }

    #[test]
    fn for_expression() {
        check(
            r#"
fn main() {
    let mut start = 0;
      //^^^^^^^^^ i32
    for increment in 0..2 { start += increment; }
      //^^^^^^^^^ i32
}"#,
        );
    }

    #[test]
    fn if_expr() {
        check(
            r#"
enum Option<T> { None, Some(T) }
use Option::*;

struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    if let None = &test {};
    if let test = &test {};
         //^^^^ &Option<Test>
    if let Some(test) = &test {};
              //^^^^ &Test
    if let Some(Test { a,             b }) = &test {};
                     //^ &Option<u32> ^ &u8
    if let Some(Test { a: x,             b: y }) = &test {};
                        //^ &Option<u32>    ^ &u8
    if let Some(Test { a: Some(x),  b: y }) = &test {};
                             //^ &u32  ^ &u8
    if let Some(Test { a: None,  b: y }) = &test {};
                                  //^ &u8
    if let Some(Test { b: y, .. }) = &test {};
                        //^ &u8
    if test == None {}
}"#,
        );
    }

    #[test]
    fn while_expr() {
        check(
            r#"
enum Option<T> { None, Some(T) }
use Option::*;

struct Test { a: Option<u32>, b: u8 }

fn main() {
    let test = Some(Test { a: Some(3), b: 1 });
      //^^^^ Option<Test>
    while let Some(Test { a: Some(x),  b: y }) = &test {};
                                //^ &u32  ^ &u8
}"#,
        );
    }

    #[test]
    fn match_arm_list() {
        check(
            r#"
enum Option<T> { None, Some(T) }
use Option::*;

struct Test { a: Option<u32>, b: u8 }

fn main() {
    match Some(Test { a: Some(3), b: 1 }) {
        None => (),
        test => (),
      //^^^^ Option<Test>
        Some(Test { a: Some(x), b: y }) => (),
                          //^ u32  ^ u8
        _ => {}
    }
}"#,
        );
    }

    #[test]
    fn hint_truncation() {
        check_with_config(
            r#"
struct Smol<T>(T);

struct VeryLongOuterName<T>(T);

fn main() {
    let a = Smol(0u32);
      //^ Smol<u32>
    let b = VeryLongOuterName(0usize);
      //^ VeryLongOuterName<…>
    let c = Smol(Smol(0u32))
      //^ Smol<Smol<…>>
}"#,
            InlayHintsConfig { max_length: Some(8), ..Default::default() },
        );
    }

    #[test]
    fn function_call_parameter_hint() {
        check(
            r#"
enum Option<T> { None, Some(T) }
use Option::*;

struct FileId {}
struct SmolStr {}

struct TextRange {}
struct SyntaxKind {}
struct NavigationTarget {}

struct Test {}

impl Test {
    fn method(&self, mut param: i32) -> i32 { param * 2 }

    fn from_syntax(
        file_id: FileId,
        name: SmolStr,
        focus_range: Option<TextRange>,
        full_range: TextRange,
        kind: SyntaxKind,
        docs: Option<String>,
    ) -> NavigationTarget {
        NavigationTarget {}
    }
}

fn test_func(mut foo: i32, bar: i32, msg: &str, _: i32, last: i32) -> i32 {
    foo + bar
}

fn main() {
    let not_literal = 1;
      //^^^^^^^^^^^ i32
    let _: i32 = test_func(1,    2,      "hello", 3,  not_literal);
                         //^ foo ^ bar   ^^^^^^^ msg  ^^^^^^^^^^^ last
    let t: Test = Test {};
    t.method(123);
           //^^^ param
    Test::method(&t,      3456);
               //^^ &self ^^^^ param
    Test::from_syntax(
        FileId {},
      //^^^^^^^^^ file_id
        "impl".into(),
      //^^^^^^^^^^^^^ name
        None,
      //^^^^ focus_range
        TextRange {},
      //^^^^^^^^^^^^ full_range
        SyntaxKind {},
      //^^^^^^^^^^^^^ kind
        None,
      //^^^^ docs
    );
}"#,
        );
    }

    #[test]
    fn omitted_parameters_hints_heuristics() {
        check_with_config(
            r#"
fn map(f: i32) {}
fn filter(predicate: i32) {}

struct TestVarContainer {
    test_var: i32,
}

impl TestVarContainer {
    fn test_var(&self) -> i32 {
        self.test_var
    }
}

struct Test {}

impl Test {
    fn map(self, f: i32) -> Self {
        self
    }

    fn filter(self, predicate: i32) -> Self {
        self
    }

    fn field(self, value: i32) -> Self {
        self
    }

    fn no_hints_expected(&self, _: i32, test_var: i32) {}

    fn frob(&self, frob: bool) {}
}

struct Param {}

fn different_order(param: &Param) {}
fn different_order_mut(param: &mut Param) {}
fn has_underscore(_param: bool) {}
fn enum_matches_param_name(completion_kind: CompletionKind) {}

fn twiddle(twiddle: bool) {}
fn doo(_doo: bool) {}

enum CompletionKind {
    Keyword,
}

fn main() {
    let container: TestVarContainer = TestVarContainer { test_var: 42 };
    let test: Test = Test {};

    map(22);
    filter(33);

    let test_processed: Test = test.map(1).filter(2).field(3);

    let test_var: i32 = 55;
    test_processed.no_hints_expected(22, test_var);
    test_processed.no_hints_expected(33, container.test_var);
    test_processed.no_hints_expected(44, container.test_var());
    test_processed.frob(false);

    twiddle(true);
    doo(true);

    let mut param_begin: Param = Param {};
    different_order(&param_begin);
    different_order(&mut param_begin);

    let param: bool = true;
    has_underscore(param);

    enum_matches_param_name(CompletionKind::Keyword);

    let a: f64 = 7.0;
    let b: f64 = 4.0;
    let _: f64 = a.div_euclid(b);
    let _: f64 = a.abs_sub(b);
}"#,
            InlayHintsConfig { max_length: Some(8), ..Default::default() },
        );
    }

    #[test]
    fn unit_structs_have_no_type_hints() {
        check_with_config(
            r#"
enum Result<T, E> { Ok(T), Err(E) }
use Result::*;

struct SyntheticSyntax;

fn main() {
    match Ok(()) {
        Ok(_) => (),
        Err(SyntheticSyntax) => (),
    }
}"#,
            InlayHintsConfig { max_length: Some(8), ..Default::default() },
        );
    }

    #[test]
    fn chaining_hints_ignore_comments() {
        check_expect(
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C))
        .into_b() // This is a comment
        .into_c();
}
"#,
            InlayHintsConfig {
                parameter_hints: false,
                type_hints: false,
                chaining_hints: true,
                max_length: None,
            },
            expect![[r#"
                [
                    InlayHint {
                        range: 147..172,
                        kind: ChainingHint,
                        label: "B",
                    },
                    InlayHint {
                        range: 147..154,
                        kind: ChainingHint,
                        label: "A",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn chaining_hints_without_newlines() {
        check_with_config(
            r#"
struct A(B);
impl A { fn into_b(self) -> B { self.0 } }
struct B(C);
impl B { fn into_c(self) -> C { self.0 } }
struct C;

fn main() {
    let c = A(B(C)).into_b().into_c();
}"#,
            InlayHintsConfig {
                parameter_hints: false,
                type_hints: false,
                chaining_hints: true,
                max_length: None,
            },
        );
    }

    #[test]
    fn struct_access_chaining_hints() {
        check_expect(
            r#"
struct A { pub b: B }
struct B { pub c: C }
struct C(pub bool);
struct D;

impl D {
    fn foo(&self) -> i32 { 42 }
}

fn main() {
    let x = A { b: B { c: C(true) } }
        .b
        .c
        .0;
    let x = D
        .foo();
}"#,
            InlayHintsConfig {
                parameter_hints: false,
                type_hints: false,
                chaining_hints: true,
                max_length: None,
            },
            expect![[r#"
                [
                    InlayHint {
                        range: 143..190,
                        kind: ChainingHint,
                        label: "C",
                    },
                    InlayHint {
                        range: 143..179,
                        kind: ChainingHint,
                        label: "B",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn generic_chaining_hints() {
        check_expect(
            r#"
struct A<T>(T);
struct B<T>(T);
struct C<T>(T);
struct X<T,R>(T, R);

impl<T> A<T> {
    fn new(t: T) -> Self { A(t) }
    fn into_b(self) -> B<T> { B(self.0) }
}
impl<T> B<T> {
    fn into_c(self) -> C<T> { C(self.0) }
}
fn main() {
    let c = A::new(X(42, true))
        .into_b()
        .into_c();
}
"#,
            InlayHintsConfig {
                parameter_hints: false,
                type_hints: false,
                chaining_hints: true,
                max_length: None,
            },
            expect![[r#"
                [
                    InlayHint {
                        range: 246..283,
                        kind: ChainingHint,
                        label: "B<X<i32, bool>>",
                    },
                    InlayHint {
                        range: 246..265,
                        kind: ChainingHint,
                        label: "A<X<i32, bool>>",
                    },
                ]
            "#]],
        );
    }
}
