use std::{iter, mem::discriminant};

use crate::{
    doc_links::token_as_doc_comment,
    navigation_target::{self, ToNav},
    FilePosition, NavigationTarget, RangeInfo, TryToNav, UpmappingResult,
};
use hir::{AsAssocItem, AssocItem, FileRange, InFile, MacroFileIdExt, ModuleDef, Semantics};
use ide_db::{
    base_db::{AnchoredPath, FileLoader, SourceDatabase},
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    RootDatabase, SymbolKind,
};
use itertools::Itertools;
use span::{Edition, FileId};
use syntax::{
    ast::{self, HasLoopBody},
    match_ast, AstNode, AstToken,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, TextRange, T,
};

// Feature: Go to Definition
//
// Navigates to the definition of an identifier.
//
// For outline modules, this will navigate to the source file of the module.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[F12]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113065563-025fbe00-91b1-11eb-83e4-a5a703610b23.gif[]
pub(crate) fn goto_definition(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = &Semantics::new(db);
    let file = sema.parse_guess_edition(file_id).syntax().clone();
    let edition =
        sema.attach_first_edition(file_id).map(|it| it.edition()).unwrap_or(Edition::CURRENT);
    let original_token = pick_best_token(file.token_at_offset(offset), |kind| match kind {
        IDENT
        | INT_NUMBER
        | LIFETIME_IDENT
        | T![self]
        | T![super]
        | T![crate]
        | T![Self]
        | COMMENT => 4,
        // index and prefix ops
        T!['['] | T![']'] | T![?] | T![*] | T![-] | T![!] => 3,
        kind if kind.is_keyword(edition) => 2,
        T!['('] | T![')'] => 2,
        kind if kind.is_trivia() => 0,
        _ => 1,
    })?;
    if let Some(doc_comment) = token_as_doc_comment(&original_token) {
        return doc_comment.get_definition_with_descend_at(sema, offset, |def, _, link_range| {
            let nav = def.try_to_nav(db)?;
            Some(RangeInfo::new(link_range, nav.collect()))
        });
    }

    if let Some((range, resolution)) =
        sema.check_for_format_args_template(original_token.clone(), offset)
    {
        return Some(RangeInfo::new(
            range,
            match resolution {
                Some(res) => def_to_nav(db, Definition::from(res)),
                None => vec![],
            },
        ));
    }

    if let Some(navs) = handle_control_flow_keywords(sema, &original_token) {
        return Some(RangeInfo::new(original_token.text_range(), navs));
    }

    let navs = sema
        .descend_into_macros_no_opaque(original_token.clone())
        .into_iter()
        .filter_map(|token| {
            let parent = token.parent()?;

            if let Some(token) = ast::String::cast(token.clone()) {
                if let Some(x) = try_lookup_include_path(sema, token, file_id) {
                    return Some(vec![x]);
                }
            }

            if ast::TokenTree::can_cast(parent.kind()) {
                if let Some(x) = try_lookup_macro_def_in_macro_use(sema, token) {
                    return Some(vec![x]);
                }
            }

            Some(
                IdentClass::classify_node(sema, &parent)?
                    .definitions()
                    .into_iter()
                    .flat_map(|def| {
                        if let Definition::ExternCrateDecl(crate_def) = def {
                            return crate_def
                                .resolved_crate(db)
                                .map(|it| it.root_module().to_nav(sema.db))
                                .into_iter()
                                .flatten()
                                .collect();
                        }
                        try_filter_trait_item_definition(sema, &def)
                            .unwrap_or_else(|| def_to_nav(sema.db, def))
                    })
                    .collect(),
            )
        })
        .flatten()
        .unique()
        .collect::<Vec<NavigationTarget>>();

    Some(RangeInfo::new(original_token.text_range(), navs))
}

fn try_lookup_include_path(
    sema: &Semantics<'_, RootDatabase>,
    token: ast::String,
    file_id: FileId,
) -> Option<NavigationTarget> {
    let file = sema.hir_file_for(&token.syntax().parent()?).macro_file()?;
    if !iter::successors(Some(file), |file| file.parent(sema.db).macro_file())
        // Check that we are in the eager argument expansion of an include macro
        .any(|file| file.is_include_like_macro(sema.db) && file.eager_arg(sema.db).is_none())
    {
        return None;
    }
    let path = token.value().ok()?;

    let file_id = sema.db.resolve_path(AnchoredPath { anchor: file_id, path: &path })?;
    let size = sema.db.file_text(file_id).len().try_into().ok()?;
    Some(NavigationTarget {
        file_id,
        full_range: TextRange::new(0.into(), size),
        name: path.into(),
        alias: None,
        focus_range: None,
        kind: None,
        container_name: None,
        description: None,
        docs: None,
    })
}

fn try_lookup_macro_def_in_macro_use(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> Option<NavigationTarget> {
    let extern_crate = token.parent()?.ancestors().find_map(ast::ExternCrate::cast)?;
    let extern_crate = sema.to_def(&extern_crate)?;
    let krate = extern_crate.resolved_crate(sema.db)?;

    for mod_def in krate.root_module().declarations(sema.db) {
        if let ModuleDef::Macro(mac) = mod_def {
            if mac.name(sema.db).as_str() == token.text() {
                if let Some(nav) = mac.try_to_nav(sema.db) {
                    return Some(nav.call_site);
                }
            }
        }
    }

    None
}

/// finds the trait definition of an impl'd item, except function
/// e.g.
/// ```rust
/// trait A { type a; }
/// struct S;
/// impl A for S { type a = i32; } // <-- on this associate type, will get the location of a in the trait
/// ```
fn try_filter_trait_item_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: &Definition,
) -> Option<Vec<NavigationTarget>> {
    let db = sema.db;
    let assoc = def.as_assoc_item(db)?;
    match assoc {
        AssocItem::Function(..) => None,
        AssocItem::Const(..) | AssocItem::TypeAlias(..) => {
            let trait_ = assoc.implemented_trait(db)?;
            let name = def.name(db)?;
            let discriminant_value = discriminant(&assoc);
            trait_
                .items(db)
                .iter()
                .filter(|itm| discriminant(*itm) == discriminant_value)
                .find_map(|itm| (itm.name(db)? == name).then(|| itm.try_to_nav(db)).flatten())
                .map(|it| it.collect())
        }
    }
}

fn handle_control_flow_keywords(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Option<Vec<NavigationTarget>> {
    match token.kind() {
        // For `fn` / `loop` / `while` / `for` / `async`, return the keyword it self,
        // so that VSCode will find the references when using `ctrl + click`
        T![fn] | T![async] | T![try] | T![return] => nav_for_exit_points(sema, token),
        T![loop] | T![while] | T![break] | T![continue] => nav_for_break_points(sema, token),
        T![for] if token.parent().and_then(ast::ForExpr::cast).is_some() => {
            nav_for_break_points(sema, token)
        }
        _ => None,
    }
}

pub(crate) fn find_fn_or_blocks(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Vec<SyntaxNode> {
    let find_ancestors = |token: SyntaxToken| {
        let token_kind = token.kind();

        for anc in sema.token_ancestors_with_macros(token) {
            let node = match_ast! {
                match anc {
                    ast::Fn(fn_) => fn_.syntax().clone(),
                    ast::ClosureExpr(c) => c.syntax().clone(),
                    ast::BlockExpr(blk) => {
                        match blk.modifier() {
                            Some(ast::BlockModifier::Async(_)) => blk.syntax().clone(),
                            Some(ast::BlockModifier::Try(_)) if token_kind != T![return] => blk.syntax().clone(),
                            _ => continue,
                        }
                    },
                    _ => continue,
                }
            };

            return Some(node);
        }
        None
    };

    sema.descend_into_macros(token.clone()).into_iter().filter_map(find_ancestors).collect_vec()
}

fn nav_for_exit_points(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Option<Vec<NavigationTarget>> {
    let db = sema.db;
    let token_kind = token.kind();

    let navs = find_fn_or_blocks(sema, token)
        .into_iter()
        .filter_map(|node| {
            let file_id = sema.hir_file_for(&node);

            match_ast! {
                match node {
                    ast::Fn(fn_) => {
                        let mut nav = sema.to_def(&fn_)?.try_to_nav(db)?;
                        // For async token, we navigate to itself, which triggers
                        // VSCode to find the references
                        let focus_token = if matches!(token_kind, T![async]) {
                            fn_.async_token()?
                        } else {
                            fn_.fn_token()?
                        };

                        let focus_frange = InFile::new(file_id, focus_token.text_range())
                            .original_node_file_range_opt(db)
                            .map(|(frange, _)| frange);

                        if let Some(FileRange { file_id, range }) = focus_frange {
                            let contains_frange = |nav: &NavigationTarget| {
                                nav.file_id == file_id && nav.full_range.contains_range(range)
                            };

                            if let Some(def_site) = nav.def_site.as_mut() {
                                if contains_frange(def_site) {
                                    def_site.focus_range = Some(range);
                                }
                            } else if contains_frange(&nav.call_site) {
                                nav.call_site.focus_range = Some(range);
                            }
                        }

                        Some(nav)
                    },
                    ast::ClosureExpr(c) => {
                        let pipe_tok = c.param_list().and_then(|it| it.pipe_token())?.text_range();
                        let closure_in_file = InFile::new(file_id, c.into());
                        Some(expr_to_nav(db, closure_in_file, Some(pipe_tok)))
                    },
                    ast::BlockExpr(blk) => {
                        match blk.modifier() {
                            Some(ast::BlockModifier::Async(_)) => {
                                let async_tok = blk.async_token()?.text_range();
                                let blk_in_file = InFile::new(file_id, blk.into());
                                Some(expr_to_nav(db, blk_in_file, Some(async_tok)))
                            },
                            Some(ast::BlockModifier::Try(_)) if token_kind != T![return] => {
                                let try_tok = blk.try_token()?.text_range();
                                let blk_in_file = InFile::new(file_id, blk.into());
                                Some(expr_to_nav(db, blk_in_file, Some(try_tok)))
                            },
                            _ => None,
                        }
                    },
                    _ => None,
                }
            }
        })
        .flatten()
        .collect_vec();

    Some(navs)
}

pub(crate) fn find_loops(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Option<Vec<ast::Expr>> {
    let parent = token.parent()?;
    let lbl = match_ast! {
        match parent {
            ast::BreakExpr(break_) => break_.lifetime(),
            ast::ContinueExpr(continue_) => continue_.lifetime(),
            _ => None,
        }
    };
    let label_matches =
        |it: Option<ast::Label>| match (lbl.as_ref(), it.and_then(|it| it.lifetime())) {
            (Some(lbl), Some(it)) => lbl.text() == it.text(),
            (None, _) => true,
            (Some(_), None) => false,
        };

    let find_ancestors = |token: SyntaxToken| {
        for anc in sema.token_ancestors_with_macros(token).filter_map(ast::Expr::cast) {
            let node = match &anc {
                ast::Expr::LoopExpr(loop_) if label_matches(loop_.label()) => anc,
                ast::Expr::WhileExpr(while_) if label_matches(while_.label()) => anc,
                ast::Expr::ForExpr(for_) if label_matches(for_.label()) => anc,
                ast::Expr::BlockExpr(blk)
                    if blk.label().is_some() && label_matches(blk.label()) =>
                {
                    anc
                }
                _ => continue,
            };

            return Some(node);
        }
        None
    };

    sema.descend_into_macros(token.clone())
        .into_iter()
        .filter_map(find_ancestors)
        .collect_vec()
        .into()
}

fn nav_for_break_points(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Option<Vec<NavigationTarget>> {
    let db = sema.db;

    let navs = find_loops(sema, token)?
        .into_iter()
        .filter_map(|expr| {
            let file_id = sema.hir_file_for(expr.syntax());
            let expr_in_file = InFile::new(file_id, expr.clone());
            let focus_range = match expr {
                ast::Expr::LoopExpr(loop_) => loop_.loop_token()?.text_range(),
                ast::Expr::WhileExpr(while_) => while_.while_token()?.text_range(),
                ast::Expr::ForExpr(for_) => for_.for_token()?.text_range(),
                // We guarantee that the label exists
                ast::Expr::BlockExpr(blk) => blk.label().unwrap().syntax().text_range(),
                _ => return None,
            };
            let nav = expr_to_nav(db, expr_in_file, Some(focus_range));
            Some(nav)
        })
        .flatten()
        .collect_vec();

    Some(navs)
}

fn def_to_nav(db: &RootDatabase, def: Definition) -> Vec<NavigationTarget> {
    def.try_to_nav(db).map(|it| it.collect()).unwrap_or_default()
}

fn expr_to_nav(
    db: &RootDatabase,
    InFile { file_id, value }: InFile<ast::Expr>,
    focus_range: Option<TextRange>,
) -> UpmappingResult<NavigationTarget> {
    let kind = SymbolKind::Label;

    let value_range = value.syntax().text_range();
    let navs = navigation_target::orig_range_with_focus_r(db, file_id, value_range, focus_range);
    navs.map(|(hir::FileRangeWrapper { file_id, range }, focus_range)| {
        NavigationTarget::from_syntax(file_id, "<expr>".into(), focus_range, range, kind)
    })
}

#[cfg(test)]
mod tests {
    use crate::fixture;
    use ide_db::FileRange;
    use itertools::Itertools;
    use syntax::SmolStr;

    #[track_caller]
    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;

        let cmp = |&FileRange { file_id, range }: &_| (file_id, range.start());
        let navs = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|(FileRange { file_id, range }, _)| FileRange { file_id, range })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();

        assert_eq!(expected, navs);
    }

    fn check_unresolved(ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;

        assert!(navs.is_empty(), "didn't expect this to resolve anywhere: {navs:?}")
    }

    fn check_name(expected_name: &str, ra_fixture: &str) {
        let (analysis, position, _) = fixture::annotations(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;
        assert!(navs.len() < 2, "expected single navigation target but encountered {}", navs.len());
        let Some(target) = navs.into_iter().next() else {
            panic!("expected single navigation target but encountered none");
        };
        assert_eq!(target.name, SmolStr::new_inline(expected_name));
    }

    #[test]
    fn goto_def_pat_range_to_inclusive() {
        check_name(
            "RangeToInclusive",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        ..$0='z' => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_pat_range_to() {
        check_name(
            "RangeTo",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        .$0.'z' => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_pat_range() {
        check_name(
            "Range",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        'a'.$0.'z' => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_pat_range_inclusive() {
        check_name(
            "RangeInclusive",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        'a'..$0='z' => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_pat_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        'a'..$0 => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_expr_range() {
        check_name(
            "Range",
            r#"
//- minicore: range
let x = 0.$0.1;
"#,
        );
    }

    #[test]
    fn goto_def_expr_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[0.$0.]
}
"#,
        );
    }

    #[test]
    fn goto_def_expr_range_inclusive() {
        check_name(
            "RangeInclusive",
            r#"
//- minicore: range
let x = 0.$0.=1;
"#,
        );
    }

    #[test]
    fn goto_def_expr_range_full() {
        check_name(
            "RangeFull",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[.$0.]
}
"#,
        );
    }

    #[test]
    fn goto_def_expr_range_to() {
        check_name(
            "RangeTo",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[.$0.10]
}
"#,
        );
    }

    #[test]
    fn goto_def_expr_range_to_inclusive() {
        check_name(
            "RangeToInclusive",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[.$0.=10]
}
"#,
        );
    }

    #[test]
    fn goto_def_in_included_file() {
        check(
            r#"
//- minicore:include
//- /main.rs

include!("a.rs");

fn main() {
    foo();
}

//- /a.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}

fn foo() {
    func_in_include$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_in_included_file_nested() {
        check(
            r#"
//- minicore:include
//- /main.rs

macro_rules! passthrough {
    ($($tt:tt)*) => { $($tt)* }
}

passthrough!(include!("a.rs"));

fn main() {
    foo();
}

//- /a.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}

fn foo() {
    func_in_include$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_in_included_file_inside_mod() {
        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("b.rs");
}
//- /b.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}
fn foo() {
    func_in_include$0();
}
"#,
        );

        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("a.rs");
}
//- /a.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}

fn foo() {
    func_in_include$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_if_items_same_name() {
        check(
            r#"
trait Trait {
    type A;
    const A: i32;
        //^
}

struct T;
impl Trait for T {
    type A = i32;
    const A$0: i32 = -9;
}"#,
        );
    }
    #[test]
    fn goto_def_in_mac_call_in_attr_invoc() {
        check(
            r#"
//- proc_macros: identity
pub struct Struct {
        // ^^^^^^
    field: i32,
}

macro_rules! identity {
    ($($tt:tt)*) => {$($tt)*};
}

#[proc_macros::identity]
fn function() {
    identity!(Struct$0 { field: 0 });
}

"#,
        )
    }

    #[test]
    fn goto_def_for_extern_crate() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std$0;
//- /std/lib.rs crate:std
// empty
//^file
"#,
        )
    }

    #[test]
    fn goto_def_for_renamed_extern_crate() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std as abc$0;
//- /std/lib.rs crate:std
// empty
//^file
"#,
        )
    }

    #[test]
    fn goto_def_in_items() {
        check(
            r#"
struct Foo;
     //^^^
enum E { X(Foo$0) }
"#,
        );
    }

    #[test]
    fn goto_def_at_start_of_item() {
        check(
            r#"
struct Foo;
     //^^^
enum E { X($0Foo) }
"#,
        );
    }

    #[test]
    fn goto_definition_resolves_correct_name() {
        check(
            r#"
//- /lib.rs
use a::Foo;
mod a;
mod b;
enum E { X(Foo$0) }

//- /a.rs
pub struct Foo;
         //^^^
//- /b.rs
pub struct Foo;
"#,
        );
    }

    #[test]
    fn goto_def_for_module_declaration() {
        check(
            r#"
//- /lib.rs
mod $0foo;

//- /foo.rs
// empty
//^file
"#,
        );

        check(
            r#"
//- /lib.rs
mod $0foo;

//- /foo/mod.rs
// empty
//^file
"#,
        );
    }

    #[test]
    fn goto_def_for_macros() {
        check(
            r#"
macro_rules! foo { () => { () } }
           //^^^
fn bar() {
    $0foo!();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_macros_from_other_crates() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo;
fn bar() {
    $0foo!();
}

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_macros_in_use_tree() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo$0;

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_with_arg() {
        check(
            r#"
//- /lib.rs
macro_rules! define_fn {
    ($name:ident) => (fn $name() {})
}

define_fn!(foo);
         //^^^

fn bar() {
   $0foo();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_no_arg() {
        check(
            r#"
//- /lib.rs
macro_rules! define_fn {
    () => (fn foo() {})
            //^^^
}

  define_fn!();
//^^^^^^^^^^^^^
fn bar() {
   $0foo();
}
"#,
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_pattern() {
        check(
            r#"
//- /lib.rs
macro_rules! foo {() => {0}}
           //^^^

fn bar() {
    match (0,1) {
        ($0foo!(), _) => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_match_arm_lhs() {
        check(
            r#"
//- /lib.rs
macro_rules! foo {() => {0}}
           //^^^
fn bar() {
    match 0 {
        $0foo!() => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_definition_works_for_consts_inside_range_pattern() {
        check(
            r#"
//- /lib.rs
const A: u32 = 0;
    //^

fn bar(v: u32) {
    match v {
        0..=$0A => {}
        _ => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_use_alias() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo as bar$0;

//- /foo/lib.rs crate:foo
// empty
//^file
"#,
        );
    }

    #[test]
    fn goto_def_for_use_alias_foo_macro() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo as bar$0;

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_methods() {
        check(
            r#"
struct Foo;
impl Foo {
    fn frobnicate(&self) { }
     //^^^^^^^^^^
}

fn bar(foo: &Foo) {
    foo.frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_fields() {
        check(
            r#"
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: &Foo) {
    foo.spam$0;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar() -> Foo {
    Foo {
        spam$0: 0,
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_pat_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: Foo) -> Foo {
    let Foo { spam$0: _, } = foo
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_fields_macros() {
        check(
            r"
macro_rules! m { () => { 92 };}
struct Foo { spam: u32 }
           //^^^^

fn bar() -> Foo {
    Foo { spam$0: m!() }
}
",
        );
    }

    #[test]
    fn goto_for_tuple_fields() {
        check(
            r#"
struct Foo(u32);
         //^^^

fn bar() {
    let foo = Foo(0);
    foo.$00;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_inherent_methods() {
        check(
            r#"
struct Foo;
impl Foo {
    fn frobnicate() { }
}    //^^^^^^^^^^

fn bar(foo: &Foo) {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_traits() {
        check(
            r#"
trait Foo {
    fn frobnicate();
}    //^^^^^^^^^^

fn bar() {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_self() {
        check(
            r#"
struct Foo;
trait Trait {
    fn frobnicate();
}    //^^^^^^^^^^
impl Trait for Foo {}

fn bar() {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_definition_on_self() {
        check(
            r#"
struct Foo;
impl Foo {
   //^^^
    pub fn new() -> Self {
        Self$0 {}
    }
}
"#,
        );
        check(
            r#"
struct Foo;
impl Foo {
   //^^^
    pub fn new() -> Self$0 {
        Self {}
    }
}
"#,
        );

        check(
            r#"
enum Foo { A }
impl Foo {
   //^^^
    pub fn new() -> Self$0 {
        Foo::A
    }
}
"#,
        );

        check(
            r#"
enum Foo { A }
impl Foo {
   //^^^
    pub fn thing(a: &Self$0) {
    }
}
"#,
        );
    }

    #[test]
    fn goto_definition_on_self_in_trait_impl() {
        check(
            r#"
struct Foo;
trait Make {
    fn new() -> Self;
}
impl Make for Foo {
            //^^^
    fn new() -> Self {
        Self$0 {}
    }
}
"#,
        );

        check(
            r#"
struct Foo;
trait Make {
    fn new() -> Self;
}
impl Make for Foo {
            //^^^
    fn new() -> Self$0 {
        Self {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_when_used_on_definition_name_itself() {
        check(
            r#"
struct Foo$0 { value: u32 }
     //^^^
            "#,
        );

        check(
            r#"
struct Foo {
    field$0: string,
} //^^^^^
"#,
        );

        check(
            r#"
fn foo_test$0() { }
 //^^^^^^^^
"#,
        );

        check(
            r#"
enum Foo$0 { Variant }
   //^^^
"#,
        );

        check(
            r#"
enum Foo {
    Variant1,
    Variant2$0,
  //^^^^^^^^
    Variant3,
}
"#,
        );

        check(
            r#"
static INNER$0: &str = "";
     //^^^^^
"#,
        );

        check(
            r#"
const INNER$0: &str = "";
    //^^^^^
"#,
        );

        check(
            r#"
type Thing$0 = Option<()>;
   //^^^^^
"#,
        );

        check(
            r#"
trait Foo$0 { }
    //^^^
"#,
        );

        check(
            r#"
trait Foo$0 = ;
    //^^^
"#,
        );

        check(
            r#"
mod bar$0 { }
  //^^^
"#,
        );
    }

    #[test]
    fn goto_from_macro() {
        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => { $($tt)* }
}
fn foo() {}
 //^^^
id! {
    fn bar() {
        fo$0o();
    }
}
mod confuse_index { fn foo(); }
"#,
        );
    }

    #[test]
    fn goto_through_format() {
        check(
            r#"
//- minicore: fmt
#[macro_export]
macro_rules! format {
    ($($arg:tt)*) => ($crate::fmt::format($crate::__export::format_args!($($arg)*)))
}
pub mod __export {
    pub use core::format_args;
    fn foo() {} // for index confusion
}
fn foo() -> i8 {}
 //^^^
fn test() {
    format!("{}", fo$0o())
}
"#,
        );
    }

    #[test]
    fn goto_through_included_file() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {}

include!("foo.rs");

fn f() {
    foo$0();
}

mod confuse_index {
    pub fn foo() {}
}

//- /foo.rs
fn foo() {}
 //^^^
        "#,
        );
    }

    #[test]
    fn goto_through_included_file_struct_with_doc_comment() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {}

include!("foo.rs");

fn f() {
    let x = Foo$0;
}

mod confuse_index {
    pub struct Foo;
}

//- /foo.rs
/// This is a doc comment
pub struct Foo;
         //^^^
        "#,
        );
    }

    #[test]
    fn goto_for_type_param() {
        check(
            r#"
struct Foo<T: Clone> { t: $0T }
         //^
"#,
        );
    }

    #[test]
    fn goto_within_macro() {
        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => ($($tt)*)
}

fn foo() {
    let x = 1;
      //^
    id!({
        let y = $0x;
        let z = y;
    });
}
"#,
        );

        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => ($($tt)*)
}

fn foo() {
    let x = 1;
    id!({
        let y = x;
          //^
        let z = $0y;
    });
}
"#,
        );
    }

    #[test]
    fn goto_def_in_local_fn() {
        check(
            r#"
fn main() {
    fn foo() {
        let x = 92;
          //^
        $0x;
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_in_local_macro() {
        check(
            r#"
fn bar() {
    macro_rules! foo { () => { () } }
               //^^^
    $0foo!();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_field_init_shorthand() {
        check(
            r#"
struct Foo { x: i32 }
           //^
fn main() {
    let x = 92;
      //^
    Foo { x$0 };
}
"#,
        )
    }

    #[test]
    fn goto_def_for_enum_variant_field() {
        check(
            r#"
enum Foo {
    Bar { x: i32 }
        //^
}
fn baz(foo: Foo) {
    match foo {
        Foo::Bar { x$0 } => x
                 //^
    };
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_pattern_const() {
        check(
            r#"
enum Foo { Bar }
         //^^^
impl Foo {
    fn baz(self) {
        match self { Self::Bar$0 => {} }
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_pattern_record() {
        check(
            r#"
enum Foo { Bar { val: i32 } }
         //^^^
impl Foo {
    fn baz(self) -> i32 {
        match self { Self::Bar$0 { val } => {} }
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_expr_const() {
        check(
            r#"
enum Foo { Bar }
         //^^^
impl Foo {
    fn baz(self) { Self::Bar$0; }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_expr_record() {
        check(
            r#"
enum Foo { Bar { val: i32 } }
         //^^^
impl Foo {
    fn baz(self) { Self::Bar$0 {val: 4}; }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_type_alias_generic_parameter() {
        check(
            r#"
type Alias<T> = T$0;
         //^
"#,
        )
    }

    #[test]
    fn goto_def_for_macro_container() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
foo::module$0::mac!();

//- /foo/lib.rs crate:foo
pub mod module {
      //^^^^^^
    #[macro_export]
    macro_rules! _mac { () => { () } }
    pub use crate::_mac as mac;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_in_path() {
        check(
            r#"
trait Iterator {
    type Item;
       //^^^^
}

fn f() -> impl Iterator<Item$0 = u8> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_super_assoc_ty_in_path() {
        check(
            r#"
trait Super {
    type Item;
       //^^^^
}

trait Sub: Super {}

fn f() -> impl Sub<Item$0 = u8> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_module_declaration_in_path_if_types_and_values_same_name() {
        check(
            r#"
mod bar {
    pub struct Foo {}
             //^^^
    pub fn Foo() {}
}

fn baz() {
    let _foo_enum: bar::Foo$0 = bar::Foo {};
}
        "#,
        )
    }

    #[test]
    fn unknown_assoc_ty() {
        check_unresolved(
            r#"
trait Iterator { type Item; }
fn f() -> impl Iterator<Invalid$0 = u8> {}
"#,
        )
    }

    #[test]
    fn goto_def_for_assoc_ty_in_path_multiple() {
        check(
            r#"
trait Iterator {
    type A;
       //^
    type B;
}

fn f() -> impl Iterator<A$0 = u8, B = ()> {}
"#,
        );
        check(
            r#"
trait Iterator {
    type A;
    type B;
       //^
}

fn f() -> impl Iterator<A = u8, B$0 = ()> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_ufcs() {
        check(
            r#"
trait Iterator {
    type Item;
       //^^^^
}

fn g() -> <() as Iterator<Item$0 = ()>>::Item {}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_ufcs_multiple() {
        check(
            r#"
trait Iterator {
    type A;
       //^
    type B;
}

fn g() -> <() as Iterator<A$0 = (), B = u8>>::B {}
"#,
        );
        check(
            r#"
trait Iterator {
    type A;
    type B;
       //^
}

fn g() -> <() as Iterator<A = (), B$0 = u8>>::A {}
"#,
        );
    }

    #[test]
    fn goto_self_param_ty_specified() {
        check(
            r#"
struct Foo {}

impl Foo {
    fn bar(self: &Foo) {
         //^^^^
        let foo = sel$0f;
    }
}"#,
        )
    }

    #[test]
    fn goto_self_param_on_decl() {
        check(
            r#"
struct Foo {}

impl Foo {
    fn bar(&self$0) {
          //^^^^
    }
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_on_decl() {
        check(
            r#"
fn foo<'foobar$0>(_: &'foobar ()) {
     //^^^^^^^
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_decl() {
        check(
            r#"
fn foo<'foobar>(_: &'foobar$0 ()) {
     //^^^^^^^
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_decl_nested() {
        check(
            r#"
fn foo<'foobar>(_: &'foobar ()) {
    fn foo<'foobar>(_: &'foobar$0 ()) {}
         //^^^^^^^
}"#,
        )
    }

    #[test]
    fn goto_lifetime_hrtb() {
        // FIXME: requires the HIR to somehow track these hrtb lifetimes
        check_unresolved(
            r#"
trait Foo<T> {}
fn foo<T>() where for<'a> T: Foo<&'a$0 (u8, u16)>, {}
                    //^^
"#,
        );
        check_unresolved(
            r#"
trait Foo<T> {}
fn foo<T>() where for<'a$0> T: Foo<&'a (u8, u16)>, {}
                    //^^
"#,
        );
    }

    #[test]
    fn goto_lifetime_hrtb_for_type() {
        // FIXME: requires ForTypes to be implemented
        check_unresolved(
            r#"trait Foo<T> {}
fn foo<T>() where T: for<'a> Foo<&'a$0 (u8, u16)>, {}
                       //^^
"#,
        );
    }

    #[test]
    fn goto_label() {
        check(
            r#"
fn foo<'foo>(_: &'foo ()) {
    'foo: {
  //^^^^
        'bar: loop {
            break 'foo$0;
        }
    }
}"#,
        )
    }

    #[test]
    fn goto_def_for_intra_doc_link_same_file() {
        check(
            r#"
/// Blah, [`bar`](bar) .. [`foo`](foo$0) has [`bar`](bar)
pub fn bar() { }

/// You might want to see [`std::fs::read()`] too.
pub fn foo() { }
     //^^^

}"#,
        )
    }

    #[test]
    fn goto_def_for_intra_doc_link_inner() {
        check(
            r#"
//- /main.rs
mod m;
struct S;
     //^

//- /m.rs
//! [`super::S$0`]
"#,
        )
    }

    #[test]
    fn goto_incomplete_field() {
        check(
            r#"
struct A { a: u32 }
         //^
fn foo() { A { a$0: }; }
"#,
        )
    }

    #[test]
    fn goto_proc_macro() {
        check(
            r#"
//- /main.rs crate:main deps:mac
use mac::fn_macro;

fn_macro$0!();

//- /mac.rs crate:mac
#![crate_type="proc-macro"]
#[proc_macro]
fn fn_macro() {}
 //^^^^^^^^
            "#,
        )
    }

    #[test]
    fn goto_intra_doc_links() {
        check(
            r#"

pub mod theitem {
    /// This is the item. Cool!
    pub struct TheItem;
             //^^^^^^^
}

/// Gives you a [`TheItem$0`].
///
/// [`TheItem`]: theitem::TheItem
pub fn gimme() -> theitem::TheItem {
    theitem::TheItem
}
"#,
        );
    }

    #[test]
    fn goto_ident_from_pat_macro() {
        check(
            r#"
macro_rules! pat {
    ($name:ident) => { Enum::Variant1($name) }
}

enum Enum {
    Variant1(u8),
    Variant2,
}

fn f(e: Enum) {
    match e {
        pat!(bind) => {
           //^^^^
            bind$0
        }
        Enum::Variant2 => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_include() {
        check(
            r#"
//- /main.rs

#[rustc_builtin_macro]
macro_rules! include_str {}

fn main() {
    let str = include_str!("foo.txt$0");
}
//- /foo.txt
// empty
//^file
"#,
        );
    }

    #[test]
    fn goto_include_has_eager_input() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include_str {}
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() {
    let str = include_str!(concat!("foo", ".tx$0t"));
}
//- /foo.txt
// empty
//^file
"#,
        );
    }

    #[test]
    fn goto_doc_include_str() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include_str {}

#[doc = include_str!("docs.md$0")]
struct Item;

//- /docs.md
// docs
//^file
"#,
        );
    }

    #[test]
    fn goto_shadow_include() {
        check(
            r#"
//- /main.rs
macro_rules! include {
    ("included.rs") => {}
}

include!("included.rs$0");

//- /included.rs
// empty
"#,
        );
    }

    mod goto_impl_of_trait_fn {
        use super::check;
        #[test]
        fn cursor_on_impl() {
            check(
                r#"
trait Twait {
    fn a();
}

struct Stwuct;

impl Twait for Stwuct {
    fn a$0();
     //^
}
        "#,
            );
        }
        #[test]
        fn method_call() {
            check(
                r#"
trait Twait {
    fn a(&self);
}

struct Stwuct;

impl Twait for Stwuct {
    fn a(&self){};
     //^
}
fn f() {
    let s = Stwuct;
    s.a$0();
}
        "#,
            );
        }
        #[test]
        fn method_call_inside_block() {
            check(
                r#"
trait Twait {
    fn a(&self);
}

fn outer() {
    struct Stwuct;

    impl Twait for Stwuct {
        fn a(&self){}
         //^
    }
    fn f() {
        let s = Stwuct;
        s.a$0();
    }
}
        "#,
            );
        }
        #[test]
        fn path_call() {
            check(
                r#"
trait Twait {
    fn a(&self);
}

struct Stwuct;

impl Twait for Stwuct {
    fn a(&self){};
     //^
}
fn f() {
    let s = Stwuct;
    Stwuct::a$0(&s);
}
        "#,
            );
        }
        #[test]
        fn where_clause_can_work() {
            check(
                r#"
trait G {
    fn g(&self);
}
trait Bound{}
trait EA{}
struct Gen<T>(T);
impl <T:EA> G for Gen<T> {
    fn g(&self) {
    }
}
impl <T> G for Gen<T>
where T : Bound
{
    fn g(&self){
     //^
    }
}
struct A;
impl Bound for A{}
fn f() {
    let gen = Gen::<A>(A);
    gen.g$0();
}
                "#,
            );
        }
        #[test]
        fn wc_case_is_ok() {
            check(
                r#"
trait G {
    fn g(&self);
}
trait BParent{}
trait Bound: BParent{}
struct Gen<T>(T);
impl <T> G for Gen<T>
where T : Bound
{
    fn g(&self){
     //^
    }
}
struct A;
impl Bound for A{}
fn f() {
    let gen = Gen::<A>(A);
    gen.g$0();
}
"#,
            );
        }

        #[test]
        fn method_call_defaulted() {
            check(
                r#"
trait Twait {
    fn a(&self) {}
     //^
}

struct Stwuct;

impl Twait for Stwuct {
}
fn f() {
    let s = Stwuct;
    s.a$0();
}
        "#,
            );
        }

        #[test]
        fn method_call_on_generic() {
            check(
                r#"
trait Twait {
    fn a(&self) {}
     //^
}

fn f<T: Twait>(s: T) {
    s.a$0();
}
        "#,
            );
        }
    }

    #[test]
    fn goto_def_of_trait_impl_const() {
        check(
            r#"
trait Twait {
    const NOMS: bool;
       // ^^^^
}

struct Stwuct;

impl Twait for Stwuct {
    const NOMS$0: bool = true;
}
"#,
        );
    }

    #[test]
    fn goto_def_of_trait_impl_type_alias() {
        check(
            r#"
trait Twait {
    type IsBad;
      // ^^^^^
}

struct Stwuct;

impl Twait for Stwuct {
    type IsBad$0 = !;
}
"#,
        );
    }

    #[test]
    fn goto_def_derive_input() {
        check(
            r#"
        //- minicore:derive
        #[rustc_builtin_macro]
        pub macro Copy {}
               // ^^^^
        #[derive(Copy$0)]
        struct Foo;
                    "#,
        );
        check(
            r#"
//- minicore:derive
#[rustc_builtin_macro]
pub macro Copy {}
       // ^^^^
#[cfg_attr(feature = "false", derive)]
#[derive(Copy$0)]
struct Foo;
            "#,
        );
        check(
            r#"
//- minicore:derive
mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
           // ^^^^
}
#[derive(foo::Copy$0)]
struct Foo;
            "#,
        );
        check(
            r#"
//- minicore:derive
mod foo {
 // ^^^
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(foo$0::Copy)]
struct Foo;
            "#,
        );
    }

    #[test]
    fn goto_def_in_macro_multi() {
        check(
            r#"
struct Foo {
    foo: ()
  //^^^
}
macro_rules! foo {
    ($ident:ident) => {
        fn $ident(Foo { $ident }: Foo) {}
    }
}
  foo!(foo$0);
     //^^^
     //^^^
"#,
        );
        check(
            r#"
fn bar() {}
 //^^^
struct bar;
     //^^^
macro_rules! foo {
    ($ident:ident) => {
        fn foo() {
            let _: $ident = $ident;
        }
    }
}

foo!(bar$0);
"#,
        );
    }

    #[test]
    fn goto_await_poll() {
        check(
            r#"
//- minicore: future

struct MyFut;

impl core::future::Future for MyFut {
    type Output = ();

    fn poll(
     //^^^^
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>
    ) -> std::task::Poll<Self::Output>
    {
        ()
    }
}

fn f() {
    MyFut.await$0;
}
"#,
        );
    }

    #[test]
    fn goto_await_into_future_poll() {
        check(
            r#"
//- minicore: future

struct Futurable;

impl core::future::IntoFuture for Futurable {
    type IntoFuture = MyFut;
}

struct MyFut;

impl core::future::Future for MyFut {
    type Output = ();

    fn poll(
     //^^^^
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>
    ) -> std::task::Poll<Self::Output>
    {
        ()
    }
}

fn f() {
    Futurable.await$0;
}
"#,
        );
    }

    #[test]
    fn goto_try_op() {
        check(
            r#"
//- minicore: try

struct Struct;

impl core::ops::Try for Struct {
    fn branch(
     //^^^^^^
        self
    ) {}
}

fn f() {
    Struct?$0;
}
"#,
        );
    }

    #[test]
    fn goto_index_op() {
        check(
            r#"
//- minicore: index

struct Struct;

impl core::ops::Index<usize> for Struct {
    fn index(
     //^^^^^
        self
    ) {}
}

fn f() {
    Struct[0]$0;
}
"#,
        );
    }

    #[test]
    fn goto_index_mut_op() {
        check(
            r#"
//- minicore: index

struct Foo;
struct Bar;

impl core::ops::Index<usize> for Foo {
    type Output = Bar;

    fn index(&self, index: usize) -> &Self::Output {}
}

impl core::ops::IndexMut<usize> for Foo {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {}
     //^^^^^^^^^
}

fn f() {
    let mut foo = Foo;
    foo[0]$0 = Bar;
}
"#,
        );
    }

    #[test]
    fn goto_prefix_op() {
        check(
            r#"
//- minicore: deref

struct Struct;

impl core::ops::Deref for Struct {
    fn deref(
     //^^^^^
        self
    ) {}
}

fn f() {
    $0*Struct;
}
"#,
        );
    }

    #[test]
    fn goto_deref_mut() {
        check(
            r#"
//- minicore: deref, deref_mut

struct Foo;
struct Bar;

impl core::ops::Deref for Foo {
    type Target = Bar;
    fn deref(&self) -> &Self::Target {}
}

impl core::ops::DerefMut for Foo {
    fn deref_mut(&mut self) -> &mut Self::Target {}
     //^^^^^^^^^
}

fn f() {
    let a = Foo;
    $0*a = Bar;
}
"#,
        );
    }

    #[test]
    fn goto_bin_op() {
        check(
            r#"
//- minicore: add

struct Struct;

impl core::ops::Add for Struct {
    fn add(
     //^^^
        self
    ) {}
}

fn f() {
    Struct +$0 Struct;
}
"#,
        );
    }

    #[test]
    fn goto_bin_op_multiple_impl() {
        check(
            r#"
//- minicore: add
struct S;
impl core::ops::Add for S {
    fn add(
     //^^^
    ) {}
}
impl core::ops::Add<usize> for S {
    fn add(
    ) {}
}

fn f() {
    S +$0 S
}
"#,
        );

        check(
            r#"
//- minicore: add
struct S;
impl core::ops::Add for S {
    fn add(
    ) {}
}
impl core::ops::Add<usize> for S {
    fn add(
     //^^^
    ) {}
}

fn f() {
    S +$0 0usize
}
"#,
        );
    }

    #[test]
    fn path_call_multiple_trait_impl() {
        check(
            r#"
trait Trait<T> {
    fn f(_: T);
}
impl Trait<i32> for usize {
    fn f(_: i32) {}
     //^
}
impl Trait<i64> for usize {
    fn f(_: i64) {}
}
fn main() {
    usize::f$0(0i32);
}
"#,
        );

        check(
            r#"
trait Trait<T> {
    fn f(_: T);
}
impl Trait<i32> for usize {
    fn f(_: i32) {}
}
impl Trait<i64> for usize {
    fn f(_: i64) {}
     //^
}
fn main() {
    usize::f$0(0i64);
}
"#,
        )
    }

    #[test]
    fn query_impls_in_nearest_block() {
        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
}
fn f1() {
    struct S1;
    impl S1 {
        fn e() -> () {}
         //^
    }
    fn f2() {
        fn f3() {
            S1::e$0();
        }
    }
}
"#,
        );

        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
}
fn f1() {
    struct S1;
    impl S1 {
        fn e() -> () {}
         //^
    }
    fn f2() {
        struct S2;
        S1::e$0();
    }
}
fn f12() {
    struct S1;
    impl S1 {
        fn e() -> () {}
    }
}
"#,
        );

        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
     //^
}
fn f2() {
    struct S2;
    S1::e$0();
}
"#,
        );
    }

    #[test]
    fn implicit_format_args() {
        check(
            r#"
//- minicore: fmt
fn test() {
    let a = "world";
     // ^
    format_args!("hello {a$0}");
}
"#,
        );
    }

    #[test]
    fn goto_macro_def_from_macro_use() {
        check(
            r#"
//- /main.rs crate:main deps:mac
#[macro_use(foo$0)]
extern crate mac;

//- /mac.rs crate:mac
#[macro_export]
macro_rules! foo {
           //^^^
    () => {};
}
            "#,
        );

        check(
            r#"
//- /main.rs crate:main deps:mac
#[macro_use(foo, bar$0, baz)]
extern crate mac;

//- /mac.rs crate:mac
#[macro_export]
macro_rules! foo {
    () => {};
}

#[macro_export]
macro_rules! bar {
           //^^^
    () => {};
}

#[macro_export]
macro_rules! baz {
    () => {};
}
            "#,
        );
    }

    #[test]
    fn goto_shadowed_preludes_in_block_module() {
        check(
            r#"
//- /main.rs crate:main edition:2021 deps:core
pub struct S;
         //^

fn main() {
    fn f() -> S$0 {
        fn inner() {} // forces a block def map
        return S;
    }
}
//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        pub enum S;
    }
}
        "#,
        );
    }

    #[test]
    fn goto_def_on_return_kw() {
        check(
            r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
 // ^^
        N!(i, 5, {
            println!("{}", i);
            return$0;
        });

        for i in 1..5 {
            return;
        }
       (|| {
            return;
        })();
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_return_kw_in_closure() {
        check(
            r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
        N!(i, 5, {
            println!("{}", i);
            return;
        });

        for i in 1..5 {
            return;
        }
       (|| {
     // ^
            return$0;
        })();
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_break_kw() {
        check(
            r#"
fn main() {
    for i in 1..5 {
 // ^^^
        break$0;
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_continue_kw() {
        check(
            r#"
fn main() {
    for i in 1..5 {
 // ^^^
        continue$0;
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_break_kw_for_block() {
        check(
            r#"
fn main() {
    'a:{
 // ^^^
        break$0 'a;
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_break_with_label() {
        check(
            r#"
fn foo() {
    'outer: loop {
         // ^^^^
         'inner: loop {
            'innermost: loop {
            }
            break$0 'outer;
        }
    }
}
"#,
        );
    }

    #[test]
    fn label_inside_macro() {
        check(
            r#"
macro_rules! m {
    ($s:stmt) => { $s };
}

fn foo() {
    'label: loop {
 // ^^^^^^
        m!(continue 'label$0);
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_on_return_in_try() {
        check(
            r#"
fn main() {
    fn f() {
 // ^^
        try {
            return$0;
        }

        return;
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_break_in_try() {
        check(
            r#"
fn main() {
    for i in 1..100 {
 // ^^^
        let x: Result<(), ()> = try {
            break$0;
        };
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_return_in_async_block() {
        check(
            r#"
fn main() {
    async {
 // ^^^^^
        return$0;
    }
}
"#,
        )
    }

    #[test]
    fn goto_def_on_for_kw() {
        check(
            r#"
fn main() {
    for$0 i in 1..5 {}
 // ^^^
}
"#,
        )
    }

    #[test]
    fn goto_def_on_fn_kw() {
        check(
            r#"
fn main() {
    fn$0 foo() {}
 // ^^
}
"#,
        )
    }

    #[test]
    fn shadow_builtin_macro() {
        check(
            r#"
//- minicore: column
//- /a.rs crate:a
#[macro_export]
macro_rules! column { () => {} }
          // ^^^^^^

//- /b.rs crate:b deps:a
use a::column;
fn foo() {
    $0column!();
}
        "#,
        );
    }

    #[test]
    fn issue_18138() {
        check(
            r#"
mod foo {
    macro_rules! x {
        () => {
            pub struct Foo;
                    // ^^^
        };
    }
    pub(crate) use x as m;
}

mod bar {
    use crate::m;

    m!();
 // ^^^^^

    fn qux() {
        Foo$0;
    }
}

mod m {}

use foo::m;
"#,
        );
    }

    #[test]
    fn macro_label_hygiene() {
        check(
            r#"
macro_rules! m {
    ($x:stmt) => {
        'bar: loop { $x }
    };
}

fn foo() {
    'bar: loop {
 // ^^^^
        m!(continue 'bar$0);
    }
}
"#,
        );
    }
}
