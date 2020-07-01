//! Entry point for call-hierarchy

use indexmap::IndexMap;

use hir::Semantics;
use ra_ide_db::RootDatabase;
use ra_syntax::{ast, match_ast, AstNode, TextRange};

use crate::{
    call_info::FnCallNode, display::ToNav, goto_definition, references, FilePosition,
    NavigationTarget, RangeInfo,
};

#[derive(Debug, Clone)]
pub struct CallItem {
    pub target: NavigationTarget,
    pub ranges: Vec<TextRange>,
}

impl CallItem {
    #[cfg(test)]
    pub(crate) fn assert_match(&self, expected: &str) {
        let actual = self.debug_render();
        test_utils::assert_eq_text!(expected.trim(), actual.trim(),);
    }

    #[cfg(test)]
    pub(crate) fn debug_render(&self) -> String {
        format!("{} : {:?}", self.target.debug_render(), self.ranges)
    }
}

pub(crate) fn call_hierarchy(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    goto_definition::goto_definition(db, position)
}

pub(crate) fn incoming_calls(db: &RootDatabase, position: FilePosition) -> Option<Vec<CallItem>> {
    let sema = Semantics::new(db);

    // 1. Find all refs
    // 2. Loop through refs and determine unique fndef. This will become our `from: CallHierarchyItem,` in the reply.
    // 3. Add ranges relative to the start of the fndef.
    let refs = references::find_all_refs(&sema, position, None)?;

    let mut calls = CallLocations::default();

    for reference in refs.info.references() {
        let file_id = reference.file_range.file_id;
        let file = sema.parse(file_id);
        let file = file.syntax();
        let token = file.token_at_offset(reference.file_range.range.start()).next()?;
        let token = sema.descend_into_macros(token);
        let syntax = token.parent();

        // This target is the containing function
        if let Some(nav) = syntax.ancestors().find_map(|node| {
            match_ast! {
                match node {
                    ast::FnDef(it) => {
                        let def = sema.to_def(&it)?;
                        Some(def.to_nav(sema.db))
                    },
                    _ => None,
                }
            }
        }) {
            let relative_range = reference.file_range.range;
            calls.add(&nav, relative_range);
        }
    }

    Some(calls.into_items())
}

pub(crate) fn outgoing_calls(db: &RootDatabase, position: FilePosition) -> Option<Vec<CallItem>> {
    let sema = Semantics::new(db);
    let file_id = position.file_id;
    let file = sema.parse(file_id);
    let file = file.syntax();
    let token = file.token_at_offset(position.offset).next()?;
    let token = sema.descend_into_macros(token);
    let syntax = token.parent();

    let mut calls = CallLocations::default();

    syntax
        .descendants()
        .filter_map(|node| FnCallNode::with_node_exact(&node))
        .filter_map(|call_node| {
            let name_ref = call_node.name_ref()?;

            if let Some(func_target) = match &call_node {
                FnCallNode::CallExpr(expr) => {
                    //FIXME: Type::as_callable is broken
                    let callable_def = sema.type_of_expr(&expr.expr()?)?.as_callable()?;
                    match callable_def {
                        hir::CallableDef::FunctionId(it) => {
                            let fn_def: hir::Function = it.into();
                            let nav = fn_def.to_nav(db);
                            Some(nav)
                        }
                        _ => None,
                    }
                }
                FnCallNode::MethodCallExpr(expr) => {
                    let function = sema.resolve_method_call(&expr)?;
                    Some(function.to_nav(db))
                }
                FnCallNode::MacroCallExpr(macro_call) => {
                    let macro_def = sema.resolve_macro_call(&macro_call)?;
                    Some(macro_def.to_nav(db))
                }
            } {
                Some((func_target, name_ref.syntax().text_range()))
            } else {
                None
            }
        })
        .for_each(|(nav, range)| calls.add(&nav, range));

    Some(calls.into_items())
}

#[derive(Default)]
struct CallLocations {
    funcs: IndexMap<NavigationTarget, Vec<TextRange>>,
}

impl CallLocations {
    fn add(&mut self, target: &NavigationTarget, range: TextRange) {
        self.funcs.entry(target.clone()).or_default().push(range);
    }

    fn into_items(self) -> Vec<CallItem> {
        self.funcs.into_iter().map(|(target, ranges)| CallItem { target, ranges }).collect()
    }
}

#[cfg(test)]
mod tests {
    use ra_db::FilePosition;

    use crate::mock_analysis::analysis_and_position;

    fn check_hierarchy(
        ra_fixture: &str,
        expected: &str,
        expected_incoming: &[&str],
        expected_outgoing: &[&str],
    ) {
        let (analysis, pos) = analysis_and_position(ra_fixture);

        let mut navs = analysis.call_hierarchy(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        nav.assert_match(expected);

        let item_pos = FilePosition { file_id: nav.file_id(), offset: nav.range().start() };
        let incoming_calls = analysis.incoming_calls(item_pos).unwrap().unwrap();
        assert_eq!(incoming_calls.len(), expected_incoming.len());

        for call in 0..incoming_calls.len() {
            incoming_calls[call].assert_match(expected_incoming[call]);
        }

        let outgoing_calls = analysis.outgoing_calls(item_pos).unwrap().unwrap();
        assert_eq!(outgoing_calls.len(), expected_outgoing.len());

        for call in 0..outgoing_calls.len() {
            outgoing_calls[call].assert_match(expected_outgoing[call]);
        }
    }

    #[test]
    fn test_call_hierarchy_on_ref() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller() {
    call<|>ee();
}
"#,
            "callee FN_DEF FileId(1) 0..14 3..9",
            &["caller FN_DEF FileId(1) 15..44 18..24 : [33..39]"],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_on_def() {
        check_hierarchy(
            r#"
//- /lib.rs
fn call<|>ee() {}
fn caller() {
    callee();
}
"#,
            "callee FN_DEF FileId(1) 0..14 3..9",
            &["caller FN_DEF FileId(1) 15..44 18..24 : [33..39]"],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_in_same_fn() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller() {
    call<|>ee();
    callee();
}
"#,
            "callee FN_DEF FileId(1) 0..14 3..9",
            &["caller FN_DEF FileId(1) 15..58 18..24 : [33..39, 47..53]"],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_in_different_fn() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller1() {
    call<|>ee();
}

fn caller2() {
    callee();
}
"#,
            "callee FN_DEF FileId(1) 0..14 3..9",
            &[
                "caller1 FN_DEF FileId(1) 15..45 18..25 : [34..40]",
                "caller2 FN_DEF FileId(1) 47..77 50..57 : [66..72]",
            ],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_in_tests_mod() {
        check_hierarchy(
            r#"
//- /lib.rs cfg:test
fn callee() {}
fn caller1() {
    call<|>ee();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_caller() {
        callee();
    }
}
"#,
            "callee FN_DEF FileId(1) 0..14 3..9",
            &[
                "caller1 FN_DEF FileId(1) 15..45 18..25 : [34..40]",
                "test_caller FN_DEF FileId(1) 95..149 110..121 : [134..140]",
            ],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_in_different_files() {
        check_hierarchy(
            r#"
//- /lib.rs
mod foo;
use foo::callee;

fn caller() {
    call<|>ee();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            "callee FN_DEF FileId(2) 0..18 7..13",
            &["caller FN_DEF FileId(1) 27..56 30..36 : [45..51]"],
            &[],
        );
    }

    #[test]
    fn test_call_hierarchy_outgoing() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn call<|>er() {
    callee();
    callee();
}
"#,
            "caller FN_DEF FileId(1) 15..58 18..24",
            &[],
            &["callee FN_DEF FileId(1) 0..14 3..9 : [33..39, 47..53]"],
        );
    }

    #[test]
    fn test_call_hierarchy_outgoing_in_different_files() {
        check_hierarchy(
            r#"
//- /lib.rs
mod foo;
use foo::callee;

fn call<|>er() {
    callee();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            "caller FN_DEF FileId(1) 27..56 30..36",
            &[],
            &["callee FN_DEF FileId(2) 0..18 7..13 : [45..51]"],
        );
    }

    #[test]
    fn test_call_hierarchy_incoming_outgoing() {
        check_hierarchy(
            r#"
//- /lib.rs
fn caller1() {
    call<|>er2();
}

fn caller2() {
    caller3();
}

fn caller3() {

}
"#,
            "caller2 FN_DEF FileId(1) 33..64 36..43",
            &["caller1 FN_DEF FileId(1) 0..31 3..10 : [19..26]"],
            &["caller3 FN_DEF FileId(1) 66..83 69..76 : [52..59]"],
        );
    }

    #[test]
    fn test_call_hierarchy_issue_5103() {
        check_hierarchy(
            r#"
fn a() {
    b()
}

fn b() {}

fn main() {
    a<|>()
}
"#,
            "a FN_DEF FileId(1) 0..18 3..4",
            &["main FN_DEF FileId(1) 31..52 34..38 : [47..48]"],
            &["b FN_DEF FileId(1) 20..29 23..24 : [13..14]"],
        );

        check_hierarchy(
            r#"
fn a() {
    b<|>()
}

fn b() {}

fn main() {
    a()
}
"#,
            "b FN_DEF FileId(1) 20..29 23..24",
            &["a FN_DEF FileId(1) 0..18 3..4 : [13..14]"],
            &[],
        );
    }
}
