//! FIXME: write short doc here
use hir::Semantics;
use ra_ide_db::RootDatabase;
use ra_syntax::{
    ast::{self, ArgListOwner},
    match_ast, AstNode, SyntaxNode, SyntaxToken,
};
use test_utils::mark;

use crate::{FilePosition, FunctionSignature};

/// Contains information about a call site. Specifically the
/// `FunctionSignature`and current parameter.
#[derive(Debug)]
pub struct CallInfo {
    pub signature: FunctionSignature,
    pub active_parameter: Option<usize>,
}

/// Computes parameter information for the given call expression.
pub(crate) fn call_info(db: &RootDatabase, position: FilePosition) -> Option<CallInfo> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id);
    let file = file.syntax();
    let token = file.token_at_offset(position.offset).next()?;
    let token = sema.descend_into_macros(token);
    call_info_for_token(&sema, token)
}

#[derive(Debug)]
pub(crate) struct ActiveParameter {
    /// FIXME: should be `Type` and `Name
    pub(crate) ty: String,
    pub(crate) name: String,
}

impl ActiveParameter {
    pub(crate) fn at(db: &RootDatabase, position: FilePosition) -> Option<Self> {
        call_info(db, position)?.into_active_parameter()
    }

    pub(crate) fn at_token(sema: &Semantics<RootDatabase>, token: SyntaxToken) -> Option<Self> {
        call_info_for_token(sema, token)?.into_active_parameter()
    }
}

fn call_info_for_token(sema: &Semantics<RootDatabase>, token: SyntaxToken) -> Option<CallInfo> {
    // Find the calling expression and it's NameRef
    let calling_node = FnCallNode::with_node(&token.parent())?;

    let signature = match &calling_node {
        FnCallNode::CallExpr(call) => {
            //FIXME: Type::as_callable is broken
            let callable_def = sema.type_of_expr(&call.expr()?)?.as_callable()?;
            match callable_def {
                hir::CallableDef::FunctionId(it) => {
                    let fn_def = it.into();
                    FunctionSignature::from_hir(sema.db, fn_def)
                }
                hir::CallableDef::StructId(it) => {
                    FunctionSignature::from_struct(sema.db, it.into())?
                }
                hir::CallableDef::EnumVariantId(it) => {
                    FunctionSignature::from_enum_variant(sema.db, it.into())?
                }
            }
        }
        FnCallNode::MethodCallExpr(method_call) => {
            let function = sema.resolve_method_call(&method_call)?;
            FunctionSignature::from_hir(sema.db, function)
        }
        FnCallNode::MacroCallExpr(macro_call) => {
            let macro_def = sema.resolve_macro_call(&macro_call)?;
            FunctionSignature::from_macro(sema.db, macro_def)?
        }
    };

    // If we have a calling expression let's find which argument we are on
    let num_params = signature.parameters.len();

    let active_parameter = match num_params {
        0 => None,
        1 if signature.has_self_param => None,
        1 => Some(0),
        _ => {
            if let Some(arg_list) = calling_node.arg_list() {
                // Number of arguments specified at the call site
                let num_args_at_callsite = arg_list.args().count();

                let arg_list_range = arg_list.syntax().text_range();
                if !arg_list_range.contains_inclusive(token.text_range().start()) {
                    mark::hit!(call_info_bad_offset);
                    return None;
                }

                let mut param = std::cmp::min(
                    num_args_at_callsite,
                    arg_list
                        .args()
                        .take_while(|arg| {
                            arg.syntax().text_range().end() <= token.text_range().start()
                        })
                        .count(),
                );

                // If we are in a method account for `self`
                if signature.has_self_param {
                    param += 1;
                }

                Some(param)
            } else {
                None
            }
        }
    };

    Some(CallInfo { signature, active_parameter })
}

#[derive(Debug)]
pub(crate) enum FnCallNode {
    CallExpr(ast::CallExpr),
    MethodCallExpr(ast::MethodCallExpr),
    MacroCallExpr(ast::MacroCall),
}

impl FnCallNode {
    fn with_node(syntax: &SyntaxNode) -> Option<FnCallNode> {
        syntax.ancestors().find_map(|node| {
            match_ast! {
                match node {
                    ast::CallExpr(it) => Some(FnCallNode::CallExpr(it)),
                    ast::MethodCallExpr(it) => {
                        let arg_list = it.arg_list()?;
                        if !arg_list.syntax().text_range().contains_range(syntax.text_range()) {
                            return None;
                        }
                        Some(FnCallNode::MethodCallExpr(it))
                    },
                    ast::MacroCall(it) => Some(FnCallNode::MacroCallExpr(it)),
                    _ => None,
                }
            }
        })
    }

    pub(crate) fn with_node_exact(node: &SyntaxNode) -> Option<FnCallNode> {
        match_ast! {
            match node {
                ast::CallExpr(it) => Some(FnCallNode::CallExpr(it)),
                ast::MethodCallExpr(it) => Some(FnCallNode::MethodCallExpr(it)),
                ast::MacroCall(it) => Some(FnCallNode::MacroCallExpr(it)),
                _ => None,
            }
        }
    }

    pub(crate) fn name_ref(&self) -> Option<ast::NameRef> {
        match self {
            FnCallNode::CallExpr(call_expr) => Some(match call_expr.expr()? {
                ast::Expr::PathExpr(path_expr) => path_expr.path()?.segment()?.name_ref()?,
                _ => return None,
            }),

            FnCallNode::MethodCallExpr(call_expr) => {
                call_expr.syntax().children().filter_map(ast::NameRef::cast).next()
            }

            FnCallNode::MacroCallExpr(call_expr) => call_expr.path()?.segment()?.name_ref(),
        }
    }

    fn arg_list(&self) -> Option<ast::ArgList> {
        match self {
            FnCallNode::CallExpr(expr) => expr.arg_list(),
            FnCallNode::MethodCallExpr(expr) => expr.arg_list(),
            FnCallNode::MacroCallExpr(_) => None,
        }
    }
}

impl CallInfo {
    fn into_active_parameter(self) -> Option<ActiveParameter> {
        let idx = self.active_parameter?;
        let ty = self.signature.parameter_types.get(idx)?.clone();
        let name = self.signature.parameter_names.get(idx)?.clone();
        let res = ActiveParameter { ty, name };
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};
    use test_utils::mark;

    use crate::mock_analysis::analysis_and_position;

    fn check(ra_fixture: &str, expect: Expect) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        let call_info = analysis.call_info(position).unwrap();
        let actual = match call_info {
            Some(call_info) => {
                let docs = match &call_info.signature.doc {
                    None => "".to_string(),
                    Some(docs) => format!("{}\n------\n", docs.as_str()),
                };
                let params = call_info
                    .signature
                    .parameters
                    .iter()
                    .enumerate()
                    .map(|(i, param)| {
                        if Some(i) == call_info.active_parameter {
                            format!("<{}>", param)
                        } else {
                            param.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}{}\n({})\n", docs, call_info.signature, params)
            }
            None => String::new(),
        };
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_fn_signature_two_args() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>3, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3<|>, ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3,<|> ); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (x: u32, <y: u32>)
            "#]],
        );
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, <|>); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (x: u32, <y: u32>)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_empty() {
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>); }
"#,
            expect![[r#"
                fn foo(x: u32, y: u32) -> u32
                (<x: u32>, y: u32)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_two_args_first_generics() {
        check(
            r#"
fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
    where T: Copy + Display, U: Debug
{ x + y }

fn bar() { foo(<|>3, ); }
"#,
            expect![[r#"
                fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
                where T: Copy + Display,
                      U: Debug
                (<x: T>, y: U)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_no_params() {
        check(
            r#"
fn foo<T>() -> T where T: Copy + Display {}
fn bar() { foo(<|>); }
"#,
            expect![[r#"
                fn foo<T>() -> T
                where T: Copy + Display
                ()
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_impl() {
        check(
            r#"
struct F; impl F { pub fn new() { F{}} }
fn bar() {let _ : F = F::new(<|>);}
"#,
            expect![[r#"
                pub fn new()
                ()
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_self() {
        check(
            r#"
struct S;
impl S { pub fn do_it(&self) {} }

fn bar() {
    let s: S = S;
    s.do_it(<|>);
}
"#,
            expect![[r#"
                pub fn do_it(&self)
                (&self)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_for_method_with_arg() {
        check(
            r#"
struct S;
impl S { pub fn do_it(&self, x: i32) {} }

fn bar() {
    let s: S = S;
    s.do_it(<|>);
}
"#,
            expect![[r#"
                pub fn do_it(&self, x: i32)
                (&self, <x: i32>)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_simple() {
        check(
            r#"
/// test
// non-doc-comment
fn foo(j: u32) -> u32 {
    j
}

fn bar() {
    let _ = foo(<|>);
}
"#,
            expect![[r#"
                test
                ------
                fn foo(j: u32) -> u32
                (<j: u32>)
            "#]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs() {
        check(
            r#"
/// Adds one to the number given.
///
/// # Examples
///
/// ```
/// let five = 5;
///
/// assert_eq!(6, my_crate::add_one(5));
/// ```
pub fn add_one(x: i32) -> i32 {
    x + 1
}

pub fn do() {
    add_one(<|>
}"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                pub fn add_one(x: i32) -> i32
                (<x: i32>)
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_impl() {
        check(
            r#"
struct addr;
impl addr {
    /// Adds one to the number given.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = 5;
    ///
    /// assert_eq!(6, my_crate::add_one(5));
    /// ```
    pub fn add_one(x: i32) -> i32 {
        x + 1
    }
}

pub fn do_it() {
    addr {};
    addr::add_one(<|>);
}
"#,
            expect![[r##"
                Adds one to the number given.

                # Examples

                ```
                let five = 5;

                assert_eq!(6, my_crate::add_one(5));
                ```
                ------
                pub fn add_one(x: i32) -> i32
                (<x: i32>)
            "##]],
        );
    }

    #[test]
    fn test_fn_signature_with_docs_from_actix() {
        check(
            r#"
struct WriteHandler<E>;

impl<E> WriteHandler<E> {
    /// Method is called when writer emits error.
    ///
    /// If this method returns `ErrorAction::Continue` writer processing
    /// continues otherwise stream processing stops.
    fn error(&mut self, err: E, ctx: &mut Self::Context) -> Running {
        Running::Stop
    }

    /// Method is called when writer finishes.
    ///
    /// By default this method stops actor's `Context`.
    fn finished(&mut self, ctx: &mut Self::Context) {
        ctx.stop()
    }
}

pub fn foo(mut r: WriteHandler<()>) {
    r.finished(<|>);
}
"#,
            expect![[r#"
                Method is called when writer finishes.

                By default this method stops actor's `Context`.
                ------
                fn finished(&mut self, ctx: &mut Self::Context)
                (&mut self, <ctx: &mut Self::Context>)
            "#]],
        );
    }

    #[test]
    fn call_info_bad_offset() {
        mark::check!(call_info_bad_offset);
        check(
            r#"
fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo <|> (3, ); }
"#,
            expect![[""]],
        );
    }

    #[test]
    fn test_nested_method_in_lambda() {
        check(
            r#"
struct Foo;
impl Foo { fn bar(&self, _: u32) { } }

fn bar(_: u32) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || foo.bar(<|>));
}
"#,
            expect![[r#"
                fn bar(&self, _: u32)
                (&self, <_: u32>)
            "#]],
        );
    }

    #[test]
    fn works_for_tuple_structs() {
        check(
            r#"
/// A cool tuple struct
struct TS(u32, i32);
fn main() {
    let s = TS(0, <|>);
}
"#,
            expect![[r#"
                A cool tuple struct
                ------
                struct TS(u32, i32) -> TS
                (u32, <i32>)
            "#]],
        );
    }

    #[test]
    fn generic_struct() {
        check(
            r#"
struct TS<T>(T);
fn main() {
    let s = TS(<|>);
}
"#,
            expect![[r#"
                struct TS<T>(T) -> TS
                (<T>)
            "#]],
        );
    }

    #[test]
    fn cant_call_named_structs() {
        check(
            r#"
struct TS { x: u32, y: i32 }
fn main() {
    let s = TS(<|>);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn works_for_enum_variants() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::A(<|>);
}
"#,
            expect![[r#"
                A Variant
                ------
                E::A(0: i32)
                (<0: i32>)
            "#]],
        );
    }

    #[test]
    fn cant_call_enum_records() {
        check(
            r#"
enum E {
    /// A Variant
    A(i32),
    /// Another
    B,
    /// And C
    C { a: i32, b: i32 }
}

fn main() {
    let a = E::C(<|>);
}
"#,
            expect![[""]],
        );
    }

    #[test]
    fn fn_signature_for_macro() {
        check(
            r#"
/// empty macro
macro_rules! foo {
    () => {}
}

fn f() {
    foo!(<|>);
}
"#,
            expect![[r#"
                empty macro
                ------
                foo!()
                ()
            "#]],
        );
    }

    #[test]
    fn fn_signature_for_call_in_macro() {
        check(
            r#"
macro_rules! id { ($($tt:tt)*) => { $($tt)* } }
fn foo() { }
id! {
    fn bar() { foo(<|>); }
}
"#,
            expect![[r#"
                fn foo()
                ()
            "#]],
        );
    }
}
