use ra_db::SyntaxDatabase;
use ra_syntax::{
    AstNode, SyntaxNode, TextUnit, TextRange,
    SyntaxKind::FN_DEF,
    ast::{self, ArgListOwner, DocCommentsOwner},
    algo::find_node_at_offset,
};

use crate::{FilePosition, CallInfo, db::RootDatabase};

/// Computes parameter information for the given call expression.
pub(crate) fn call_info(db: &RootDatabase, position: FilePosition) -> Option<CallInfo> {
    let file = db.source_file(position.file_id);
    let syntax = file.syntax();

    // Find the calling expression and it's NameRef
    let calling_node = FnCallNode::with_node(syntax, position.offset)?;
    let name_ref = calling_node.name_ref()?;

    // Resolve the function's NameRef (NOTE: this isn't entirely accurate).
    let file_symbols = db.index_resolve(name_ref);
    let symbol = file_symbols
        .into_iter()
        .find(|it| it.ptr.kind() == FN_DEF)?;
    let fn_file = db.source_file(symbol.file_id);
    let fn_def = symbol.ptr.resolve(&fn_file);
    let fn_def = ast::FnDef::cast(&fn_def).unwrap();
    let mut call_info = CallInfo::new(fn_def)?;
    // If we have a calling expression let's find which argument we are on
    let num_params = call_info.parameters.len();
    let has_self = fn_def.param_list().and_then(|l| l.self_param()).is_some();

    if num_params == 1 {
        if !has_self {
            call_info.active_parameter = Some(0);
        }
    } else if num_params > 1 {
        // Count how many parameters into the call we are.
        // TODO: This is best effort for now and should be fixed at some point.
        // It may be better to see where we are in the arg_list and then check
        // where offset is in that list (or beyond).
        // Revisit this after we get documentation comments in.
        if let Some(ref arg_list) = calling_node.arg_list() {
            let start = arg_list.syntax().range().start();

            let range_search = TextRange::from_to(start, position.offset);
            let mut commas: usize = arg_list
                .syntax()
                .text()
                .slice(range_search)
                .to_string()
                .matches(',')
                .count();

            // If we have a method call eat the first param since it's just self.
            if has_self {
                commas += 1;
            }

            call_info.active_parameter = Some(commas);
        }
    }

    Some(call_info)
}

enum FnCallNode<'a> {
    CallExpr(&'a ast::CallExpr),
    MethodCallExpr(&'a ast::MethodCallExpr),
}

impl<'a> FnCallNode<'a> {
    pub fn with_node(syntax: &'a SyntaxNode, offset: TextUnit) -> Option<FnCallNode<'a>> {
        if let Some(expr) = find_node_at_offset::<ast::CallExpr>(syntax, offset) {
            return Some(FnCallNode::CallExpr(expr));
        }
        if let Some(expr) = find_node_at_offset::<ast::MethodCallExpr>(syntax, offset) {
            return Some(FnCallNode::MethodCallExpr(expr));
        }
        None
    }

    pub fn name_ref(&self) -> Option<&'a ast::NameRef> {
        match *self {
            FnCallNode::CallExpr(call_expr) => Some(match call_expr.expr()?.kind() {
                ast::ExprKind::PathExpr(path_expr) => path_expr.path()?.segment()?.name_ref()?,
                _ => return None,
            }),

            FnCallNode::MethodCallExpr(call_expr) => call_expr
                .syntax()
                .children()
                .filter_map(ast::NameRef::cast)
                .nth(0),
        }
    }

    pub fn arg_list(&self) -> Option<&'a ast::ArgList> {
        match *self {
            FnCallNode::CallExpr(expr) => expr.arg_list(),
            FnCallNode::MethodCallExpr(expr) => expr.arg_list(),
        }
    }
}

impl CallInfo {
    fn new(node: &ast::FnDef) -> Option<Self> {
        let label: String = if let Some(body) = node.body() {
            let body_range = body.syntax().range();
            let label: String = node
                .syntax()
                .children()
                .filter(|child| !child.range().is_subrange(&body_range)) // Filter out body
                .filter(|child| ast::Comment::cast(child).is_none()) // Filter out doc comments
                .map(|node| node.text().to_string())
                .collect();
            label
        } else {
            node.syntax().text().to_string()
        };

        let mut doc = None;
        let docs = node.doc_comment_text();
        if !docs.is_empty() {
            // Massage markdown
            let mut processed_lines = Vec::new();
            let mut in_code_block = false;
            for line in docs.lines() {
                if line.starts_with("```") {
                    in_code_block = !in_code_block;
                }

                let line = if in_code_block && line.starts_with("```") && !line.contains("rust") {
                    "```rust".into()
                } else {
                    line.to_string()
                };

                processed_lines.push(line);
            }

            doc = Some(processed_lines.join("\n"));
        }

        Some(CallInfo {
            parameters: param_list(node),
            label: label.trim().to_owned(),
            doc,
            active_parameter: None,
        })
    }
}

fn param_list(node: &ast::FnDef) -> Vec<String> {
    let mut res = vec![];
    if let Some(param_list) = node.param_list() {
        if let Some(self_param) = param_list.self_param() {
            res.push(self_param.syntax().text().to_string())
        }

        // Maybe use param.pat here? See if we can just extract the name?
        //res.extend(param_list.params().map(|p| p.syntax().text().to_string()));
        res.extend(
            param_list
                .params()
                .filter_map(|p| p.pat())
                .map(|pat| pat.syntax().text().to_string()),
        );
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mock_analysis::single_file_with_position;

    fn call_info(text: &str) -> CallInfo {
        let (analysis, position) = single_file_with_position(text);
        analysis.call_info(position).unwrap().unwrap()
    }

    #[test]
    fn test_fn_signature_two_args_first() {
        let info = call_info(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(<|>3, ); }"#,
        );

        assert_eq!(info.parameters, vec!("x".to_string(), "y".to_string()));
        assert_eq!(info.active_parameter, Some(0));
    }

    #[test]
    fn test_fn_signature_two_args_second() {
        let info = call_info(
            r#"fn foo(x: u32, y: u32) -> u32 {x + y}
fn bar() { foo(3, <|>); }"#,
        );

        assert_eq!(info.parameters, vec!("x".to_string(), "y".to_string()));
        assert_eq!(info.active_parameter, Some(1));
    }

    #[test]
    fn test_fn_signature_for_impl() {
        let info = call_info(
            r#"struct F; impl F { pub fn new() { F{}} }
fn bar() {let _ : F = F::new(<|>);}"#,
        );

        assert_eq!(info.parameters, Vec::<String>::new());
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

        assert_eq!(info.parameters, vec!["&self".to_string()]);
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

        assert_eq!(info.parameters, vec!["&self".to_string(), "x".to_string()]);
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

        assert_eq!(info.parameters, vec!["j".to_string()]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label, "fn foo(j: u32) -> u32".to_string());
        assert_eq!(info.doc, Some("test".into()));
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

        assert_eq!(info.parameters, vec!["x".to_string()]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label, "pub fn add_one(x: i32) -> i32".to_string());
        assert_eq!(
            info.doc,
            Some(
                r#"Adds one to the number given.

# Examples

```rust
let five = 5;

assert_eq!(6, my_crate::add_one(5));
```"#
                    .into()
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

        assert_eq!(info.parameters, vec!["x".to_string()]);
        assert_eq!(info.active_parameter, Some(0));
        assert_eq!(info.label, "pub fn add_one(x: i32) -> i32".to_string());
        assert_eq!(
            info.doc,
            Some(
                r#"Adds one to the number given.

# Examples

```rust
let five = 5;

assert_eq!(6, my_crate::add_one(5));
```"#
                    .into()
            )
        );
    }

    #[test]
    fn test_fn_signature_with_docs_from_actix() {
        let info = call_info(
            r#"
pub trait WriteHandler<E>
where
    Self: Actor,
    Self::Context: ActorContext,
{
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

pub fn foo() {
    WriteHandler r;
    r.finished(<|>);
}

"#,
        );

        assert_eq!(
            info.parameters,
            vec!["&mut self".to_string(), "ctx".to_string()]
        );
        assert_eq!(info.active_parameter, Some(1));
        assert_eq!(
            info.doc,
            Some(
                r#"Method is called when writer finishes.

By default this method stops actor's `Context`."#
                    .into()
            )
        );
    }

}
