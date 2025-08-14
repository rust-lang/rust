use std::{iter, mem::discriminant};

use crate::{
    FilePosition, NavigationTarget, RangeInfo, TryToNav, UpmappingResult,
    doc_links::token_as_doc_comment,
    navigation_target::{self, ToNav},
};
use hir::{
    AsAssocItem, AssocItem, CallableKind, FileRange, HasCrate, InFile, ModuleDef, Semantics, sym,
};
use ide_db::{
    RootDatabase, SymbolKind,
    base_db::{AnchoredPath, SourceDatabase},
    defs::{Definition, IdentClass},
    famous_defs::FamousDefs,
    helpers::pick_best_token,
};
use itertools::Itertools;
use span::{Edition, FileId};
use syntax::{
    AstNode, AstToken,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, T, TextRange,
    ast::{self, HasLoopBody},
    match_ast,
};

// Feature: Go to Definition
//
// Navigates to the definition of an identifier.
//
// For outline modules, this will navigate to the source file of the module.
//
// | Editor  | Shortcut |
// |---------|----------|
// | VS Code | <kbd>F12</kbd> |
//
// ![Go to Definition](https://user-images.githubusercontent.com/48062697/113065563-025fbe00-91b1-11eb-83e4-a5a703610b23.gif)
pub(crate) fn goto_definition(
    db: &RootDatabase,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = &Semantics::new(db);
    let file = sema.parse_guess_edition(file_id).syntax().clone();
    let edition =
        sema.attach_first_edition(file_id).map(|it| it.edition(db)).unwrap_or(Edition::CURRENT);
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

    if let Some((range, _, _, resolution)) =
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

    if let Some(navs) = find_definition_for_known_blanket_dual_impls(sema, &original_token) {
        return Some(RangeInfo::new(original_token.text_range(), navs));
    }

    let navs = sema
        .descend_into_macros_no_opaque(original_token.clone(), false)
        .into_iter()
        .filter_map(|token| {
            let parent = token.value.parent()?;

            let token_file_id = token.file_id;
            if let Some(token) = ast::String::cast(token.value.clone())
                && let Some(x) =
                    try_lookup_include_path(sema, InFile::new(token_file_id, token), file_id)
            {
                return Some(vec![x]);
            }

            if ast::TokenTree::can_cast(parent.kind())
                && let Some(x) = try_lookup_macro_def_in_macro_use(sema, token.value)
            {
                return Some(vec![x]);
            }

            Some(
                IdentClass::classify_node(sema, &parent)?
                    .definitions()
                    .into_iter()
                    .flat_map(|(def, _)| {
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

// If the token is into(), try_into(), search the definition of From, TryFrom.
fn find_definition_for_known_blanket_dual_impls(
    sema: &Semantics<'_, RootDatabase>,
    original_token: &SyntaxToken,
) -> Option<Vec<NavigationTarget>> {
    let method_call = ast::MethodCallExpr::cast(original_token.parent()?.parent()?)?;
    let callable = sema.resolve_method_call_as_callable(&method_call)?;
    let CallableKind::Function(f) = callable.kind() else { return None };
    let assoc = f.as_assoc_item(sema.db)?;

    let return_type = callable.return_type();
    let fd = FamousDefs(sema, return_type.krate(sema.db));

    let t = match assoc.container(sema.db) {
        hir::AssocItemContainer::Trait(t) => t,
        hir::AssocItemContainer::Impl(impl_)
            if impl_.self_ty(sema.db).is_str() && f.name(sema.db) == sym::parse =>
        {
            let t = fd.core_convert_FromStr()?;
            let t_f = t.function(sema.db, &sym::from_str)?;
            return sema
                .resolve_trait_impl_method(
                    return_type.clone(),
                    t,
                    t_f,
                    [return_type.type_arguments().next()?],
                )
                .map(|f| def_to_nav(sema.db, f.into()));
        }
        hir::AssocItemContainer::Impl(_) => return None,
    };

    let fn_name = f.name(sema.db);
    let f = if fn_name == sym::into && fd.core_convert_Into() == Some(t) {
        let dual = fd.core_convert_From()?;
        let dual_f = dual.function(sema.db, &sym::from)?;
        sema.resolve_trait_impl_method(
            return_type.clone(),
            dual,
            dual_f,
            [return_type, callable.receiver_param(sema.db)?.1],
        )?
    } else if fn_name == sym::try_into && fd.core_convert_TryInto() == Some(t) {
        let dual = fd.core_convert_TryFrom()?;
        let dual_f = dual.function(sema.db, &sym::try_from)?;
        sema.resolve_trait_impl_method(
            return_type.clone(),
            dual,
            dual_f,
            // Extract the `T` from `Result<T, ..>`
            [return_type.type_arguments().next()?, callable.receiver_param(sema.db)?.1],
        )?
    } else if fn_name == sym::to_string && fd.alloc_string_ToString() == Some(t) {
        let dual = fd.core_fmt_Display()?;
        let dual_f = dual.function(sema.db, &sym::fmt)?;
        sema.resolve_trait_impl_method(
            return_type.clone(),
            dual,
            dual_f,
            [callable.receiver_param(sema.db)?.1.strip_reference()],
        )?
    } else {
        return None;
    };
    // Assert that we got a trait impl function, if we are back in a trait definition we didn't
    // succeed
    let _t = f.as_assoc_item(sema.db)?.implemented_trait(sema.db)?;
    let def = Definition::from(f);
    Some(def_to_nav(sema.db, def))
}

fn try_lookup_include_path(
    sema: &Semantics<'_, RootDatabase>,
    token: InFile<ast::String>,
    file_id: FileId,
) -> Option<NavigationTarget> {
    let file = token.file_id.macro_file()?;

    // Check that we are in the eager argument expansion of an include macro
    // that is we are the string input of it
    if !iter::successors(Some(file), |file| file.parent(sema.db).macro_file())
        .any(|file| file.is_include_like_macro(sema.db) && file.eager_arg(sema.db).is_none())
    {
        return None;
    }
    let path = token.value.value().ok()?;

    let file_id = sema.db.resolve_path(AnchoredPath { anchor: file_id, path: &path })?;
    let size = sema.db.file_text(file_id).text(sema.db).len().try_into().ok()?;
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
        if let ModuleDef::Macro(mac) = mod_def
            && mac.name(sema.db).as_str() == token.text()
            && let Some(nav) = mac.try_to_nav(sema.db)
        {
            return Some(nav.call_site);
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
        // For `fn` / `loop` / `while` / `for` / `async` / `match`, return the keyword it self,
        // so that VSCode will find the references when using `ctrl + click`
        T![fn] | T![async] | T![try] | T![return] => nav_for_exit_points(sema, token),
        T![loop] | T![while] | T![break] | T![continue] => nav_for_break_points(sema, token),
        T![for] if token.parent().and_then(ast::ForExpr::cast).is_some() => {
            nav_for_break_points(sema, token)
        }
        T![match] | T![=>] | T![if] => nav_for_branch_exit_points(sema, token),
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
                                nav.file_id == file_id.file_id(db) && nav.full_range.contains_range(range)
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

pub(crate) fn find_branch_root(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Vec<SyntaxNode> {
    let find_nodes = |node_filter: fn(SyntaxNode) -> Option<SyntaxNode>| {
        sema.descend_into_macros(token.clone())
            .into_iter()
            .filter_map(|token| node_filter(token.parent()?))
            .collect_vec()
    };

    match token.kind() {
        T![match] => find_nodes(|node| Some(ast::MatchExpr::cast(node)?.syntax().clone())),
        T![=>] => find_nodes(|node| Some(ast::MatchArm::cast(node)?.syntax().clone())),
        T![if] => find_nodes(|node| {
            let if_expr = ast::IfExpr::cast(node)?;

            let root_if = iter::successors(Some(if_expr.clone()), |if_expr| {
                let parent_if = if_expr.syntax().parent().and_then(ast::IfExpr::cast)?;
                let ast::ElseBranch::IfExpr(else_branch) = parent_if.else_branch()? else {
                    return None;
                };

                (else_branch.syntax() == if_expr.syntax()).then_some(parent_if)
            })
            .last()?;

            Some(root_if.syntax().clone())
        }),
        _ => vec![],
    }
}

fn nav_for_branch_exit_points(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
) -> Option<Vec<NavigationTarget>> {
    let db = sema.db;

    let navs = match token.kind() {
        T![match] => find_branch_root(sema, token)
            .into_iter()
            .filter_map(|node| {
                let file_id = sema.hir_file_for(&node);
                let match_expr = ast::MatchExpr::cast(node)?;
                let focus_range = match_expr.match_token()?.text_range();
                let match_expr_in_file = InFile::new(file_id, match_expr.into());
                Some(expr_to_nav(db, match_expr_in_file, Some(focus_range)))
            })
            .flatten()
            .collect_vec(),

        T![=>] => find_branch_root(sema, token)
            .into_iter()
            .filter_map(|node| {
                let match_arm = ast::MatchArm::cast(node)?;
                let match_expr = sema
                    .ancestors_with_macros(match_arm.syntax().clone())
                    .find_map(ast::MatchExpr::cast)?;
                let file_id = sema.hir_file_for(match_expr.syntax());
                let focus_range = match_arm.fat_arrow_token()?.text_range();
                let match_expr_in_file = InFile::new(file_id, match_expr.into());
                Some(expr_to_nav(db, match_expr_in_file, Some(focus_range)))
            })
            .flatten()
            .collect_vec(),

        T![if] => find_branch_root(sema, token)
            .into_iter()
            .filter_map(|node| {
                let file_id = sema.hir_file_for(&node);
                let if_expr = ast::IfExpr::cast(node)?;
                let focus_range = if_expr.if_token()?.text_range();
                let if_expr_in_file = InFile::new(file_id, if_expr.into());
                Some(expr_to_nav(db, if_expr_in_file, Some(focus_range)))
            })
            .flatten()
            .collect_vec(),

        _ => return Some(Vec::new()),
    };

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
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
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

    fn check_unresolved(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;

        assert!(navs.is_empty(), "didn't expect this to resolve anywhere: {navs:?}")
    }

    fn check_name(expected_name: &str, #[rust_analyzer::rust_fixture] ra_fixture: &str) {
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
//^^^^^^^^^^
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
    fn goto_def_for_intra_doc_link_outer_same_file() {
        check(
            r#"
/// [`S$0`]
mod m {
    //! [`super::S`]
}
struct S;
     //^
            "#,
        );

        check(
            r#"
/// [`S$0`]
mod m {}
struct S;
     //^
            "#,
        );

        check(
            r#"
/// [`S$0`]
fn f() {
    //! [`S`]
}
struct S;
     //^
            "#,
        );
    }

    #[test]
    fn goto_def_for_intra_doc_link_inner_same_file() {
        check(
            r#"
/// [`S`]
mod m {
    //! [`super::S$0`]
}
struct S;
     //^
            "#,
        );

        check(
            r#"
mod m {
    //! [`super::S$0`]
}
struct S;
     //^
            "#,
        );

        check(
            r#"
fn f() {
    //! [`S$0`]
}
struct S;
     //^
            "#,
        );
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

    // macros in this position are not yet supported
    #[test]
    // FIXME
    #[should_panic]
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
    let g = Gen::<A>(A);
    g.g$0();
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
    let g = Gen::<A>(A);
    g.g$0();
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
 // ^^

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
    #[test]
    fn into_call_to_from_definition() {
        check(
            r#"
//- minicore: from
struct A;

struct B;

impl From<A> for B {
    fn from(value: A) -> Self {
     //^^^^
        B
    }
}

fn f() {
    let a = A;
    let b: B = a.into$0();
}
        "#,
        );
    }

    #[test]
    fn into_call_to_from_definition_with_trait_bounds() {
        check(
            r#"
//- minicore: from, iterator
struct A;

impl<T> From<T> for A
where
    T: IntoIterator<Item = i64>,
{
    fn from(value: T) -> Self {
     //^^^^
        A
    }
}

fn f() {
    let a: A = [1, 2, 3].into$0();
}
        "#,
        );
    }

    #[test]
    fn goto_into_definition_if_exists() {
        check(
            r#"
//- minicore: from
struct A;

struct B;

impl Into<B> for A {
    fn into(self) -> B {
     //^^^^
        B
    }
}

fn f() {
    let a = A;
    let b: B = a.into$0();
}
        "#,
        );
    }

    #[test]
    fn try_into_call_to_try_from_definition() {
        check(
            r#"
//- minicore: from
struct A;

struct B;

impl TryFrom<A> for B {
    type Error = String;

    fn try_from(value: A) -> Result<Self, Self::Error> {
     //^^^^^^^^
        Ok(B)
    }
}

fn f() {
    let a = A;
    let b: Result<B, _> = a.try_into$0();
}
        "#,
        );
    }

    #[test]
    fn goto_try_into_definition_if_exists() {
        check(
            r#"
//- minicore: from
struct A;

struct B;

impl TryInto<B> for A {
    type Error = String;

    fn try_into(self) -> Result<B, Self::Error> {
     //^^^^^^^^
        Ok(B)
    }
}

fn f() {
    let a = A;
    let b: Result<B, _> = a.try_into$0();
}
        "#,
        );
    }

    #[test]
    fn parse_call_to_from_str_definition() {
        check(
            r#"
//- minicore: from, str
struct A;
impl FromStr for A {
    type Error = String;
    fn from_str(value: &str) -> Result<Self, Self::Error> {
     //^^^^^^^^
        Ok(A)
    }
}
fn f() {
    let a: Result<A, _> = "aaaaaa".parse$0();
}
        "#,
        );
    }

    #[test]
    fn to_string_call_to_display_definition() {
        check(
            r#"
//- minicore:fmt
//- /alloc.rs crate:alloc
pub mod string {
    pub struct String;
    pub trait ToString {
        fn to_string(&self) -> String;
    }

    impl<T: core::fmt::Display> ToString for T {
        fn to_string(&self) -> String { String }
    }
}
//- /lib.rs crate:lib deps:alloc
use alloc::string::ToString;
struct A;
impl core::fmt::Display for A {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {}
    // ^^^
}
fn f() {
    A.to_string$0();
}
        "#,
        );
    }

    #[test]
    fn use_inside_body() {
        check(
            r#"
fn main() {
    mod nice_module {
        pub(super) struct NiceStruct;
                       // ^^^^^^^^^^
    }

    use nice_module::NiceStruct$0;

    let _ = NiceStruct;
}
    "#,
        );
    }

    #[test]
    fn shadow_builtin_type_by_module() {
        check(
            r#"
mod Foo{
pub mod str {
     // ^^^
    pub fn foo() {}
}
}

fn main() {
    use Foo::str;
    let s = st$0r::foo();
}
"#,
        );
    }

    #[test]
    fn not_goto_module_because_str_is_builtin_type() {
        check(
            r#"
mod str {
pub fn foo() {}
}

fn main() {
    let s = st$0r::f();
}
"#,
        );
    }

    #[test]
    fn struct_shadow_by_module() {
        check(
            r#"
mod foo {
    pub mod bar {
         // ^^^
        pub type baz = usize;
    }
}
struct bar;
fn main() {
    use foo::bar;
    let x: ba$0r::baz = 5;

}
"#,
        );
    }

    #[test]
    fn type_alias_shadow_by_module() {
        check(
            r#"
mod foo {
    pub mod bar {
         // ^^^
        pub fn baz() {}
    }
}

trait Qux {}

fn item<bar: Qux>() {
    use foo::bar;
    ba$0r::baz();
}
}
"#,
        );

        check(
            r#"
mod foo {
    pub mod bar {
         // ^^^
        pub fn baz() {}
    }
}

fn item<bar>(x: bar) {
    use foo::bar;
    let x: bar$0 = x;
}
"#,
        );
    }

    #[test]
    fn trait_shadow_by_module() {
        check(
            r#"
pub mod foo {
    pub mod Bar {}
         // ^^^
}

trait Bar {}

fn main() {
    use foo::Bar;
    fn f<Qux: B$0ar>() {}
}
            "#,
        );
    }

    #[test]
    fn const_shadow_by_module() {
        check(
            r#"
pub mod foo {
    pub struct u8 {}
    pub mod bar {
        pub mod u8 {}
    }
}

fn main() {
    use foo::u8;
    {
        use foo::bar::u8;

        fn f1<const N: u$08>() {}
    }
    fn f2<const N: u8>() {}
}
"#,
        );

        check(
            r#"
pub mod foo {
    pub struct u8 {}
            // ^^
    pub mod bar {
        pub mod u8 {}
    }
}

fn main() {
    use foo::u8;
    {
        use foo::bar::u8;

        fn f1<const N: u8>() {}
    }
    fn f2<const N: u$08>() {}
}
"#,
        );

        check(
            r#"
pub mod foo {
    pub struct buz {}
    pub mod bar {
        pub mod buz {}
             // ^^^
    }
}

fn main() {
    use foo::buz;
    {
        use foo::bar::buz;

        fn f1<const N: buz$0>() {}
    }
}
"#,
        );
    }

    #[test]
    fn offset_of() {
        check(
            r#"
//- minicore: offset_of
struct Foo {
    field: i32,
 // ^^^^^
}

fn foo() {
    let _ = core::mem::offset_of!(Foo, fiel$0d);
}
        "#,
        );

        check(
            r#"
//- minicore: offset_of
struct Bar(Foo);
struct Foo {
    field: i32,
 // ^^^^^
}

fn foo() {
    let _ = core::mem::offset_of!(Bar, 0.fiel$0d);
}
        "#,
        );

        check(
            r#"
//- minicore: offset_of
struct Bar(Baz);
enum Baz {
    Abc(Foo),
    None,
}
struct Foo {
    field: i32,
 // ^^^^^
}

fn foo() {
    let _ = core::mem::offset_of!(Bar, 0.Abc.0.fiel$0d);
}
        "#,
        );

        check(
            r#"
//- minicore: offset_of
struct Bar(Baz);
enum Baz {
    Abc(Foo),
 // ^^^
    None,
}
struct Foo {
    field: i32,
}

fn foo() {
    let _ = core::mem::offset_of!(Bar, 0.Ab$0c.0.field);
}
        "#,
        );
    }

    #[test]
    fn goto_def_for_match_keyword() {
        check(
            r#"
fn main() {
    match$0 0 {
 // ^^^^^
        0 => {},
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_match_arm_fat_arrow() {
        check(
            r#"
fn main() {
    match 0 {
        0 =>$0 {},
       // ^^
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_if_keyword() {
        check(
            r#"
fn main() {
    if$0 true {
 // ^^
        ()
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_match_nested_in_if() {
        check(
            r#"
fn main() {
    if true {
        match$0 0 {
     // ^^^^^
            0 => {},
            _ => {},
        }
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_multiple_match_expressions() {
        check(
            r#"
fn main() {
    match 0 {
        0 => {},
        _ => {},
    };

    match$0 1 {
 // ^^^^^
        1 => {},
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_nested_match_expressions() {
        check(
            r#"
fn main() {
    match 0 {
        0 => match$0 1 {
          // ^^^^^
            1 => {},
            _ => {},
        },
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_if_else_chains() {
        check(
            r#"
fn main() {
    if true {
 // ^^
        ()
    } else if$0 false {
        ()
    } else {
        ()
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_match_with_guards() {
        check(
            r#"
fn main() {
    match 42 {
        x if x > 0 =>$0 {},
                // ^^
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_match_with_macro_arm() {
        check(
            r#"
macro_rules! arm {
    () => { 0 => {} };
}

fn main() {
    match$0 0 {
 // ^^^^^
        arm!(),
        _ => {},
    }
}
"#,
        );
    }

    #[test]
    fn goto_const_from_match_pat_with_tuple_struct() {
        check(
            r#"
struct Tag(u8);
struct Path {}

const Path: u8 = 0;
   // ^^^^
fn main() {
    match Tag(Path) {
        Tag(Path$0) => {}
        _ => {}
    }
}

"#,
        );
    }

    #[test]
    fn goto_const_from_match_pat() {
        check(
            r#"
type T1 = u8;
const T1: u8 = 0;
   // ^^
fn main() {
    let x = 0;
    match x {
        T1$0 => {}
        _ => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_struct_from_match_pat() {
        check(
            r#"
struct T1;
    // ^^
fn main() {
    let x = 0;
    match x {
        T1$0 => {}
        _ => {}
    }
}
"#,
        );
    }

    #[test]
    fn no_goto_trait_from_match_pat() {
        check(
            r#"
trait T1 {}
fn main() {
    let x = 0;
    match x {
        T1$0 => {}
     // ^^
        _ => {}
    }
}
"#,
        );
    }
}
