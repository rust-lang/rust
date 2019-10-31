//! FIXME: write short doc here

use ra_db::SourceDatabase;
use ra_syntax::{
    algo::ancestors_at_offset,
    ast::{self, ArgListOwner},
    match_ast, AstNode, SyntaxNode, TextUnit,
};
use test_utils::tested_by;

use crate::{db::RootDatabase, CallInfo, FilePosition, FunctionSignature};

/// Computes parameter information for the given call expression.
pub(crate) fn call_info(db: &RootDatabase, position: FilePosition) -> Option<CallInfo> {
    let parse = db.parse(position.file_id);
    let syntax = parse.tree().syntax().clone();

    // Find the calling expression and it's NameRef
    let calling_node = FnCallNode::with_node(&syntax, position.offset)?;
    let name_ref = calling_node.name_ref()?;

    let analyzer = hir::SourceAnalyzer::new(db, position.file_id, name_ref.syntax(), None);
    let (mut call_info, has_self) = match &calling_node {
        FnCallNode::CallExpr(expr) => {
            //FIXME: apply subst
            let (callable_def, _subst) = analyzer.type_of(db, &expr.expr()?)?.as_callable()?;
            match callable_def {
                hir::CallableDef::Function(it) => {
                    (CallInfo::with_fn(db, it), it.data(db).has_self_param())
                }
                hir::CallableDef::Struct(it) => (CallInfo::with_struct(db, it)?, false),
                hir::CallableDef::EnumVariant(it) => (CallInfo::with_enum_variant(db, it)?, false),
            }
        }
        FnCallNode::MethodCallExpr(expr) => {
            let function = analyzer.resolve_method_call(&expr)?;
            (CallInfo::with_fn(db, function), function.data(db).has_self_param())
        }
        FnCallNode::MacroCallExpr(expr) => {
            let macro_def = analyzer.resolve_macro_call(db, &expr)?;
            (CallInfo::with_macro(db, macro_def)?, false)
        }
    };

    // If we have a calling expression let's find which argument we are on
    let num_params = call_info.parameters().len();

    if num_params == 1 {
        if !has_self {
            call_info.active_parameter = Some(0);
        }
    } else if num_params > 1 {
        // Count how many parameters into the call we are.
        if let Some(arg_list) = calling_node.arg_list() {
            // Number of arguments specified at the call site
            let num_args_at_callsite = arg_list.args().count();

            let arg_list_range = arg_list.syntax().text_range();
            if !arg_list_range.contains_inclusive(position.offset) {
                tested_by!(call_info_bad_offset);
                return None;
            }

            let mut param = std::cmp::min(
                num_args_at_callsite,
                arg_list
                    .args()
                    .take_while(|arg| arg.syntax().text_range().end() < position.offset)
                    .count(),
            );

            // If we are in a method account for `self`
            if has_self {
                param += 1;
            }

            call_info.active_parameter = Some(param);
        }
    }

    Some(call_info)
}

#[derive(Debug)]
enum FnCallNode {
    CallExpr(ast::CallExpr),
    MethodCallExpr(ast::MethodCallExpr),
    MacroCallExpr(ast::MacroCall),
}

impl FnCallNode {
    fn with_node(syntax: &SyntaxNode, offset: TextUnit) -> Option<FnCallNode> {
        ancestors_at_offset(syntax, offset).find_map(|node| {
            match_ast! {
                match node {
                    ast::CallExpr(it) => { Some(FnCallNode::CallExpr(it)) },
                    ast::MethodCallExpr(it) => { Some(FnCallNode::MethodCallExpr(it)) },
                    ast::MacroCall(it) => { Some(FnCallNode::MacroCallExpr(it)) },
                    _ => { None },
                }
            }
        })
    }

    fn name_ref(&self) -> Option<ast::NameRef> {
        match self {
            FnCallNode::CallExpr(call_expr) => Some(match call_expr.expr()? {
                ast::Expr::PathExpr(path_expr) => path_expr.path()?.segment()?.name_ref()?,
                _ => return None,
            }),

            FnCallNode::MethodCallExpr(call_expr) => {
                call_expr.syntax().children().filter_map(ast::NameRef::cast).nth(0)
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
    fn with_fn(db: &RootDatabase, function: hir::Function) -> Self {
        let signature = FunctionSignature::from_hir(db, function);

        CallInfo { signature, active_parameter: None }
    }

    fn with_struct(db: &RootDatabase, st: hir::Struct) -> Option<Self> {
        let signature = FunctionSignature::from_struct(db, st)?;

        Some(CallInfo { signature, active_parameter: None })
    }

    fn with_enum_variant(db: &RootDatabase, variant: hir::EnumVariant) -> Option<Self> {
        let signature = FunctionSignature::from_enum_variant(db, variant)?;

        Some(CallInfo { signature, active_parameter: None })
    }

    fn with_macro(db: &RootDatabase, macro_def: hir::MacroDef) -> Option<Self> {
        let signature = FunctionSignature::from_macro(db, macro_def)?;

        Some(CallInfo { signature, active_parameter: None })
    }

    fn parameters(&self) -> &[String] {
        &self.signature.parameters
    }
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::mock_analysis::single_file_with_position;

    use super::*;

    // These are only used when testing
    impl CallInfo {
        fn doc(&self) -> Option<hir::Documentation> {
            self.signature.doc.clone()
        }

        fn label(&self) -> String {
            self.signature.to_string()
        }
    }

    fn call_info(text: &str) -> CallInfo {
        let (analysis, position) = single_file_with_position(text);
        analysis.call_info(position).unwrap().unwrap()
    }

    #[test]
    fn test_fn_signature_two_args_firstx() {
        let info = call_info(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>3, ); }"#,
        );

        assert_eq!(info.parameters(), ["x: u32", "y: u32"]);
        assert_eq!(info.active_parameter, Some(0));
    }

    #[test]
    fn test_fn_signature_two_args_second() {
        let info = call_info(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, <|>); }"#,
        );

        assert_eq!(info.parameters(), ["x: u32", "y: u32"]);
        assert_eq!(info.active_parameter, Some(1));
    }

    #[test]
    fn test_fn_signature_two_args_empty() {
        let info = call_info(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>); }"#,
        );

        assert_eq!(info.parameters(), ["x: u32", "y: u32"]);
        assert_eq!(info.active_parameter, Some(0));
    }

    #[test]
    fn test_fn_signature_two_args_first_generics() {
        let info = call_info(
            r#"fn foo<T, U: Copy + Display>(x: T, y: U) -> u32 where T: Copy + Display, U: Debug {x + y}
fn bar() { foo(<|>3, ); }"#,
        );

        assert_eq!(info.parameters(), ["x: T", "y: U"]);
        assert_eq!(
            info.label(),
            r#"
fn foo<T, U: Copy + Display>(x: T, y: U) -> u32
where T: Copy + Display,
      U: Debug
    "#
            .trim()
        );
        assert_eq!(info.active_parameter, Some(0));
    }

    #[test]
    fn test_fn_signature_no_params() {
        let info = call_info(
            r#"fn foo<T>() -> T where T: Copy + Display {}
fn bar() { foo(<|>); }"#,
        );

        assert!(info.parameters().is_empty());
        assert_eq!(
            info.label(),
            r#"
fn foo<T>() -> T
where T: Copy + Display
    "#
            .trim()
        );
        assert!(info.active_parameter.is_none());
    }

    #[test]
    fn test_fn_signature_for_impl() {
        let info = call_info(
            r#"struct F; impl F { pub fn new() { F{}} }
fn bar() {let _ : F = F::new(<|>);}"#,
        );

        assert!(info.parameters().is_empty());
        assert_eq!(info.active_parameter, None);
    }

    #[test]
    fn test_fn_signature_for_method_self() {
        let info = call_info(
            r#"struct F;
impl F {
    pub fn new() -> F{
        F{}
    }

    pub fn do_it(&self) {}
}

fn bar() {
    let f : F = F::new();
    f.do_it(<|>);
}"#,
        );

        assert_eq!(info.parameters(), ["&self"]);
        assert_eq!(info.active_parameter, None);
    }

    #[test]
    fn test_fn_signature_for_method_with_arg() {
        let info = call_info(
            r#"struct F;
impl F {
    pub fn new() -> F{
        F{}
    }

    pub fn do_it(&self, x: i32) {}
}

fn bar() {
    let f : F = F::new();
    f.do_it(<|>);
}"#,
        );

        assert_eq!(info.parameters(), ["&self", "x: i32"]);
        assert_eq!(info.active_parameter, Some(1));
    }

    #[test]
    fn test_fn_signature_with_docs_simple() {
        let info = call_info(
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
        );

        assert_eq!(info.parameters(), ["j: u32"]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label(), "fn foo(j: u32) -> u32");
        assert_eq!(info.doc().map(|it| it.into()), Some("test".to_string()));
    }

    #[test]
    fn test_fn_signature_with_docs() {
        let info = call_info(
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
        );

        assert_eq!(info.parameters(), ["x: i32"]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label(), "pub fn add_one(x: i32) -> i32");
        assert_eq!(
            info.doc().map(|it| it.into()),
            Some(
                r#"Adds one to the number given.

# Examples

```
let five = 5;

assert_eq!(6, my_crate::add_one(5));
```"#
                    .to_string()
            )
        );
    }

    #[test]
    fn test_fn_signature_with_docs_impl() {
        let info = call_info(
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
}"#,
        );

        assert_eq!(info.parameters(), ["x: i32"]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label(), "pub fn add_one(x: i32) -> i32");
        assert_eq!(
            info.doc().map(|it| it.into()),
            Some(
                r#"Adds one to the number given.

# Examples

```
let five = 5;

assert_eq!(6, my_crate::add_one(5));
```"#
                    .to_string()
            )
        );
    }

    #[test]
    fn test_fn_signature_with_docs_from_actix() {
        let info = call_info(
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
        );

        assert_eq!(info.label(), "fn finished(&mut self, ctx: &mut Self::Context)".to_string());
        assert_eq!(info.parameters(), ["&mut self", "ctx: &mut Self::Context"]);
        assert_eq!(info.active_parameter, Some(1));
        assert_eq!(
            info.doc().map(|it| it.into()),
            Some(
                r#"Method is called when writer finishes.

By default this method stops actor's `Context`."#
                    .to_string()
            )
        );
    }

    #[test]
    fn call_info_bad_offset() {
        covers!(call_info_bad_offset);
        let (analysis, position) = single_file_with_position(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
               fn bar() { foo <|> (3, ); }"#,
        );
        let call_info = analysis.call_info(position).unwrap();
        assert!(call_info.is_none());
    }

    #[test]
    fn test_nested_method_in_lamba() {
        let info = call_info(
            r#"struct Foo;

impl Foo {
    fn bar(&self, _: u32) { }
}

fn bar(_: u32) { }

fn main() {
    let foo = Foo;
    std::thread::spawn(move || foo.bar(<|>));
}"#,
        );

        assert_eq!(info.parameters(), ["&self", "_: u32"]);
        assert_eq!(info.active_parameter, Some(1));
        assert_eq!(info.label(), "fn bar(&self, _: u32)");
    }

    #[test]
    fn works_for_tuple_structs() {
        let info = call_info(
            r#"
/// A cool tuple struct
struct TS(u32, i32);
fn main() {
    let s = TS(0, <|>);
}"#,
        );

        assert_eq!(info.label(), "struct TS(u32, i32) -> TS");
        assert_eq!(info.doc().map(|it| it.into()), Some("A cool tuple struct".to_string()));
        assert_eq!(info.active_parameter, Some(1));
    }

    #[test]
    #[should_panic]
    fn cant_call_named_structs() {
        let _ = call_info(
            r#"
struct TS { x: u32, y: i32 }
fn main() {
    let s = TS(<|>);
}"#,
        );
    }

    #[test]
    fn works_for_enum_variants() {
        let info = call_info(
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
        );

        assert_eq!(info.label(), "E::A(0: i32)");
        assert_eq!(info.doc().map(|it| it.into()), Some("A Variant".to_string()));
        assert_eq!(info.active_parameter, Some(0));
    }

    #[test]
    #[should_panic]
    fn cant_call_enum_records() {
        let _ = call_info(
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
        );
    }

    #[test]
    fn fn_signature_for_macro() {
        let info = call_info(
            r#"
/// empty macro
macro_rules! foo {
    () => {}
}

fn f() {
    foo!(<|>);
}
        "#,
        );

        assert_eq!(info.label(), "foo!()");
        assert_eq!(info.doc().map(|it| it.into()), Some("empty macro".to_string()));
    }
}
