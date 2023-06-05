//! Entry point for call-hierarchy

use hir::Semantics;
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    helpers::pick_best_token,
    search::FileReference,
    FxIndexMap, RootDatabase,
};
use syntax::{ast, AstNode, SyntaxKind::IDENT, TextRange};

use crate::{goto_definition, FilePosition, NavigationTarget, RangeInfo, TryToNav};

#[derive(Debug, Clone)]
pub struct CallItem {
    pub target: NavigationTarget,
    pub ranges: Vec<TextRange>,
}

impl CallItem {
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

pub(crate) fn incoming_calls(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<Vec<CallItem>> {
    let sema = &Semantics::new(db);

    let file = sema.parse(file_id);
    let file = file.syntax();
    let mut calls = CallLocations::default();

    let references = sema
        .find_nodes_at_offset_with_descend(file, offset)
        .filter_map(move |node| match node {
            ast::NameLike::NameRef(name_ref) => match NameRefClass::classify(sema, &name_ref)? {
                NameRefClass::Definition(def @ Definition::Function(_)) => Some(def),
                _ => None,
            },
            ast::NameLike::Name(name) => match NameClass::classify(sema, &name)? {
                NameClass::Definition(def @ Definition::Function(_)) => Some(def),
                _ => None,
            },
            ast::NameLike::Lifetime(_) => None,
        })
        .flat_map(|func| func.usages(sema).all());

    for (_, references) in references {
        let references =
            references.iter().filter_map(|FileReference { name, .. }| name.as_name_ref());
        for name in references {
            // This target is the containing function
            let nav = sema.ancestors_with_macros(name.syntax().clone()).find_map(|node| {
                let def = ast::Fn::cast(node).and_then(|fn_| sema.to_def(&fn_))?;
                def.try_to_nav(sema.db)
            });
            if let Some(nav) = nav {
                calls.add(nav, sema.original_range(name.syntax()).range);
            }
        }
    }

    Some(calls.into_items())
}

pub(crate) fn outgoing_calls(db: &RootDatabase, position: FilePosition) -> Option<Vec<CallItem>> {
    let sema = Semantics::new(db);
    let file_id = position.file_id;
    let file = sema.parse(file_id);
    let file = file.syntax();
    let token = pick_best_token(file.token_at_offset(position.offset), |kind| match kind {
        IDENT => 1,
        _ => 0,
    })?;
    let mut calls = CallLocations::default();

    sema.descend_into_macros(token)
        .into_iter()
        .filter_map(|it| it.parent_ancestors().nth(1).and_then(ast::Item::cast))
        .filter_map(|item| match item {
            ast::Item::Const(c) => c.body().map(|it| it.syntax().descendants()),
            ast::Item::Fn(f) => f.body().map(|it| it.syntax().descendants()),
            ast::Item::Static(s) => s.body().map(|it| it.syntax().descendants()),
            _ => None,
        })
        .flatten()
        .filter_map(ast::CallableExpr::cast)
        .filter_map(|call_node| {
            let (nav_target, range) = match call_node {
                ast::CallableExpr::Call(call) => {
                    let expr = call.expr()?;
                    let callable = sema.type_of_expr(&expr)?.original.as_callable(db)?;
                    match callable.kind() {
                        hir::CallableKind::Function(it) => {
                            let range = expr.syntax().text_range();
                            it.try_to_nav(db).zip(Some(range))
                        }
                        _ => None,
                    }
                }
                ast::CallableExpr::MethodCall(expr) => {
                    let range = expr.name_ref()?.syntax().text_range();
                    let function = sema.resolve_method_call(&expr)?;
                    function.try_to_nav(db).zip(Some(range))
                }
            }?;
            Some((nav_target, range))
        })
        .for_each(|(nav, range)| calls.add(nav, range));

    Some(calls.into_items())
}

#[derive(Default)]
struct CallLocations {
    funcs: FxIndexMap<NavigationTarget, Vec<TextRange>>,
}

impl CallLocations {
    fn add(&mut self, target: NavigationTarget, range: TextRange) {
        self.funcs.entry(target).or_default().push(range);
    }

    fn into_items(self) -> Vec<CallItem> {
        self.funcs.into_iter().map(|(target, ranges)| CallItem { target, ranges }).collect()
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::base_db::FilePosition;
    use itertools::Itertools;

    use crate::fixture;

    fn check_hierarchy(
        ra_fixture: &str,
        expected: Expect,
        expected_incoming: Expect,
        expected_outgoing: Expect,
    ) {
        let (analysis, pos) = fixture::position(ra_fixture);

        let mut navs = analysis.call_hierarchy(pos).unwrap().unwrap().info;
        assert_eq!(navs.len(), 1);
        let nav = navs.pop().unwrap();
        expected.assert_eq(&nav.debug_render());

        let item_pos =
            FilePosition { file_id: nav.file_id, offset: nav.focus_or_full_range().start() };
        let incoming_calls = analysis.incoming_calls(item_pos).unwrap().unwrap();
        expected_incoming
            .assert_eq(&incoming_calls.into_iter().map(|call| call.debug_render()).join("\n"));

        let outgoing_calls = analysis.outgoing_calls(item_pos).unwrap().unwrap();
        expected_outgoing
            .assert_eq(&outgoing_calls.into_iter().map(|call| call.debug_render()).join("\n"));
    }

    #[test]
    fn test_call_hierarchy_on_ref() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller() {
    call$0ee();
}
"#,
            expect![["callee Function FileId(0) 0..14 3..9"]],
            expect![["caller Function FileId(0) 15..44 18..24 : [33..39]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_on_def() {
        check_hierarchy(
            r#"
//- /lib.rs
fn call$0ee() {}
fn caller() {
    callee();
}
"#,
            expect![["callee Function FileId(0) 0..14 3..9"]],
            expect![["caller Function FileId(0) 15..44 18..24 : [33..39]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_in_same_fn() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller() {
    call$0ee();
    callee();
}
"#,
            expect![["callee Function FileId(0) 0..14 3..9"]],
            expect![["caller Function FileId(0) 15..58 18..24 : [33..39, 47..53]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_in_different_fn() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn caller1() {
    call$0ee();
}

fn caller2() {
    callee();
}
"#,
            expect![["callee Function FileId(0) 0..14 3..9"]],
            expect![["
                caller1 Function FileId(0) 15..45 18..25 : [34..40]
                caller2 Function FileId(0) 47..77 50..57 : [66..72]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_in_tests_mod() {
        check_hierarchy(
            r#"
//- /lib.rs cfg:test
fn callee() {}
fn caller1() {
    call$0ee();
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
            expect![["callee Function FileId(0) 0..14 3..9"]],
            expect![[r#"
                caller1 Function FileId(0) 15..45 18..25 : [34..40]
                test_caller Function FileId(0) 95..149 110..121 tests : [134..140]"#]],
            expect![[]],
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
    call$0ee();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            expect!["callee Function FileId(1) 0..18 7..13 foo"],
            expect![["caller Function FileId(0) 27..56 30..36 : [45..51]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_outgoing() {
        check_hierarchy(
            r#"
//- /lib.rs
fn callee() {}
fn call$0er() {
    callee();
    callee();
}
"#,
            expect![["caller Function FileId(0) 15..58 18..24"]],
            expect![[]],
            expect![["callee Function FileId(0) 0..14 3..9 : [33..39, 47..53]"]],
        );
    }

    #[test]
    fn test_call_hierarchy_outgoing_in_different_files() {
        check_hierarchy(
            r#"
//- /lib.rs
mod foo;
use foo::callee;

fn call$0er() {
    callee();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            expect![["caller Function FileId(0) 27..56 30..36"]],
            expect![[]],
            expect!["callee Function FileId(1) 0..18 7..13 foo : [45..51]"],
        );
    }

    #[test]
    fn test_call_hierarchy_incoming_outgoing() {
        check_hierarchy(
            r#"
//- /lib.rs
fn caller1() {
    call$0er2();
}

fn caller2() {
    caller3();
}

fn caller3() {

}
"#,
            expect![["caller2 Function FileId(0) 33..64 36..43"]],
            expect![["caller1 Function FileId(0) 0..31 3..10 : [19..26]"]],
            expect![["caller3 Function FileId(0) 66..83 69..76 : [52..59]"]],
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
    a$0()
}
"#,
            expect![["a Function FileId(0) 0..18 3..4"]],
            expect![["main Function FileId(0) 31..52 34..38 : [47..48]"]],
            expect![["b Function FileId(0) 20..29 23..24 : [13..14]"]],
        );

        check_hierarchy(
            r#"
fn a() {
    b$0()
}

fn b() {}

fn main() {
    a()
}
"#,
            expect![["b Function FileId(0) 20..29 23..24"]],
            expect![["a Function FileId(0) 0..18 3..4 : [13..14]"]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_in_macros_incoming() {
        check_hierarchy(
            r#"
macro_rules! define {
    ($ident:ident) => {
        fn $ident {}
    }
}
macro_rules! call {
    ($ident:ident) => {
        $ident()
    }
}
define!(callee)
fn caller() {
    call!(call$0ee);
}
"#,
            expect![[r#"callee Function FileId(0) 144..159 152..158"#]],
            expect![[r#"caller Function FileId(0) 160..194 163..169 : [184..190]"#]],
            expect![[]],
        );
        check_hierarchy(
            r#"
macro_rules! define {
    ($ident:ident) => {
        fn $ident {}
    }
}
macro_rules! call {
    ($ident:ident) => {
        $ident()
    }
}
define!(cal$0lee)
fn caller() {
    call!(callee);
}
"#,
            expect![[r#"callee Function FileId(0) 144..159 152..158"#]],
            expect![[r#"caller Function FileId(0) 160..194 163..169 : [184..190]"#]],
            expect![[]],
        );
    }

    #[test]
    fn test_call_hierarchy_in_macros_outgoing() {
        check_hierarchy(
            r#"
macro_rules! define {
    ($ident:ident) => {
        fn $ident {}
    }
}
macro_rules! call {
    ($ident:ident) => {
        $ident()
    }
}
define!(callee)
fn caller$0() {
    call!(callee);
}
"#,
            expect![[r#"caller Function FileId(0) 160..194 163..169"#]],
            expect![[]],
            // FIXME
            expect![[]],
        );
    }

    #[test]
    fn test_trait_method_call_hierarchy() {
        check_hierarchy(
            r#"
trait T1 {
    fn call$0ee();
}

struct S1;

impl T1 for S1 {
    fn callee() {}
}

fn caller() {
    S1::callee();
}
"#,
            expect!["callee Function FileId(0) 15..27 18..24 T1"],
            expect![["caller Function FileId(0) 82..115 85..91 : [104..110]"]],
            expect![[]],
        );
    }
}
