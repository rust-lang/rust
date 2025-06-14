//! Implementation of "lifetime elision" inlay hints:
//! ```no_run
//! fn example/* <'0> */(a: &/* '0 */()) {}
//! ```
use std::iter;

use ide_db::{FxHashMap, famous_defs::FamousDefs, syntax_helpers::node_ext::walk_ty};
use itertools::Itertools;
use syntax::{SmolStr, format_smolstr};
use syntax::{
    SyntaxKind, SyntaxToken,
    ast::{self, AstNode, HasGenericParams, HasName},
};

use crate::{
    InlayHint, InlayHintPosition, InlayHintsConfig, InlayKind, LifetimeElisionHints,
    inlay_hints::InlayHintCtx,
};

pub(super) fn fn_hints(
    acc: &mut Vec<InlayHint>,
    ctx: &mut InlayHintCtx,
    fd: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    func: ast::Fn,
) -> Option<()> {
    if config.lifetime_elision_hints == LifetimeElisionHints::Never {
        return None;
    }

    let param_list = func.param_list()?;
    let generic_param_list = func.generic_param_list();
    let ret_type = func.ret_type();
    let self_param = param_list.self_param().filter(|it| it.amp_token().is_some());
    let gpl_append_range = func.name()?.syntax().text_range();
    hints_(
        acc,
        ctx,
        fd,
        config,
        param_list.params().filter_map(|it| {
            Some((
                it.pat().and_then(|it| match it {
                    ast::Pat::IdentPat(p) => p.name(),
                    _ => None,
                }),
                it.ty()?,
            ))
        }),
        generic_param_list,
        ret_type,
        self_param,
        |acc, allocated_lifetimes| {
            acc.push(InlayHint {
                range: gpl_append_range,
                kind: InlayKind::GenericParamList,
                label: format!("<{}>", allocated_lifetimes.iter().format(", "),).into(),
                text_edit: None,
                position: InlayHintPosition::After,
                pad_left: false,
                pad_right: false,
                resolve_parent: None,
            })
        },
        true,
    )
}

pub(super) fn fn_ptr_hints(
    acc: &mut Vec<InlayHint>,
    ctx: &mut InlayHintCtx,
    fd: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    func: ast::FnPtrType,
) -> Option<()> {
    if config.lifetime_elision_hints == LifetimeElisionHints::Never {
        return None;
    }

    let parent_for_type = func
        .syntax()
        .ancestors()
        .skip(1)
        .take_while(|it| matches!(it.kind(), SyntaxKind::PAREN_TYPE | SyntaxKind::FOR_TYPE))
        .find_map(ast::ForType::cast);

    let param_list = func.param_list()?;
    let generic_param_list = parent_for_type.as_ref().and_then(|it| it.generic_param_list());
    let ret_type = func.ret_type();
    let for_kw = parent_for_type.as_ref().and_then(|it| it.for_token());
    hints_(
        acc,
        ctx,
        fd,
        config,
        param_list.params().filter_map(|it| {
            Some((
                it.pat().and_then(|it| match it {
                    ast::Pat::IdentPat(p) => p.name(),
                    _ => None,
                }),
                it.ty()?,
            ))
        }),
        generic_param_list,
        ret_type,
        None,
        |acc, allocated_lifetimes| {
            let has_for = for_kw.is_some();
            let for_ = if has_for { "" } else { "for" };
            acc.push(InlayHint {
                range: for_kw.map_or_else(
                    || func.syntax().first_token().unwrap().text_range(),
                    |it| it.text_range(),
                ),
                kind: InlayKind::GenericParamList,
                label: format!("{for_}<{}>", allocated_lifetimes.iter().format(", "),).into(),
                text_edit: None,
                position: if has_for {
                    InlayHintPosition::After
                } else {
                    InlayHintPosition::Before
                },
                pad_left: false,
                pad_right: true,
                resolve_parent: None,
            });
        },
        false,
    )
}

pub(super) fn fn_path_hints(
    acc: &mut Vec<InlayHint>,
    ctx: &mut InlayHintCtx,
    fd: &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    func: &ast::PathType,
) -> Option<()> {
    if config.lifetime_elision_hints == LifetimeElisionHints::Never {
        return None;
    }

    // FIXME: Support general path types
    let (param_list, ret_type) = func.path().as_ref().and_then(path_as_fn)?;
    let parent_for_type = func
        .syntax()
        .ancestors()
        .skip(1)
        .take_while(|it| matches!(it.kind(), SyntaxKind::PAREN_TYPE | SyntaxKind::FOR_TYPE))
        .find_map(ast::ForType::cast);

    let generic_param_list = parent_for_type.as_ref().and_then(|it| it.generic_param_list());
    let for_kw = parent_for_type.as_ref().and_then(|it| it.for_token());
    hints_(
        acc,
        ctx,
        fd,
        config,
        param_list.type_args().filter_map(|it| Some((None, it.ty()?))),
        generic_param_list,
        ret_type,
        None,
        |acc, allocated_lifetimes| {
            let has_for = for_kw.is_some();
            let for_ = if has_for { "" } else { "for" };
            acc.push(InlayHint {
                range: for_kw.map_or_else(
                    || func.syntax().first_token().unwrap().text_range(),
                    |it| it.text_range(),
                ),
                kind: InlayKind::GenericParamList,
                label: format!("{for_}<{}>", allocated_lifetimes.iter().format(", "),).into(),
                text_edit: None,
                position: if has_for {
                    InlayHintPosition::After
                } else {
                    InlayHintPosition::Before
                },
                pad_left: false,
                pad_right: true,
                resolve_parent: None,
            });
        },
        false,
    )
}

fn path_as_fn(path: &ast::Path) -> Option<(ast::ParenthesizedArgList, Option<ast::RetType>)> {
    path.segment().and_then(|it| it.parenthesized_arg_list().zip(Some(it.ret_type())))
}

fn hints_(
    acc: &mut Vec<InlayHint>,
    ctx: &mut InlayHintCtx,
    FamousDefs(_, _): &FamousDefs<'_, '_>,
    config: &InlayHintsConfig,
    params: impl Iterator<Item = (Option<ast::Name>, ast::Type)>,
    generic_param_list: Option<ast::GenericParamList>,
    ret_type: Option<ast::RetType>,
    self_param: Option<ast::SelfParam>,
    on_missing_gpl: impl FnOnce(&mut Vec<InlayHint>, &[SmolStr]),
    mut is_trivial: bool,
) -> Option<()> {
    let is_elided = |lt: &Option<ast::Lifetime>| match lt {
        Some(lt) => matches!(lt.text().as_str(), "'_"),
        None => true,
    };

    let mk_lt_hint = |t: SyntaxToken, label: String| InlayHint {
        range: t.text_range(),
        kind: InlayKind::Lifetime,
        label: label.into(),
        text_edit: None,
        position: InlayHintPosition::After,
        pad_left: false,
        pad_right: true,
        resolve_parent: None,
    };

    let potential_lt_refs = {
        let mut acc: Vec<_> = vec![];
        if let Some(self_param) = &self_param {
            let lifetime = self_param.lifetime();
            let is_elided = is_elided(&lifetime);
            acc.push((None, self_param.amp_token(), lifetime, is_elided));
        }
        params.for_each(|(name, ty)| {
            // FIXME: check path types
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(r) => {
                    let lifetime = r.lifetime();
                    let is_elided = is_elided(&lifetime);
                    acc.push((name.clone(), r.amp_token(), lifetime, is_elided));
                    false
                }
                ast::Type::FnPtrType(_) => {
                    is_trivial = false;
                    true
                }
                ast::Type::PathType(t) => {
                    if t.path()
                        .and_then(|it| it.segment())
                        .and_then(|it| it.parenthesized_arg_list())
                        .is_some()
                    {
                        is_trivial = false;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            })
        });
        acc
    };

    let mut used_names: FxHashMap<SmolStr, usize> =
        ctx.lifetime_stacks.iter().flat_map(|it| it.iter()).cloned().zip(iter::repeat(0)).collect();
    // allocate names
    let mut gen_idx_name = {
        let mut generic = (0u8..).map(|idx| match idx {
            idx if idx < 10 => SmolStr::from_iter(['\'', (idx + 48) as char]),
            idx => format_smolstr!("'{idx}"),
        });
        let ctx = &*ctx;
        move || {
            generic
                .by_ref()
                .find(|s| ctx.lifetime_stacks.iter().flat_map(|it| it.iter()).all(|n| n != s))
                .unwrap_or_default()
        }
    };
    let mut allocated_lifetimes = vec![];

    {
        let mut potential_lt_refs = potential_lt_refs.iter().filter(|&&(.., is_elided)| is_elided);
        if self_param.is_some() && potential_lt_refs.next().is_some() {
            allocated_lifetimes.push(if config.param_names_for_lifetime_elision_hints {
                // self can't be used as a lifetime, so no need to check for collisions
                "'self".into()
            } else {
                gen_idx_name()
            });
        }
        potential_lt_refs.for_each(|(name, ..)| {
            let name = match name {
                Some(it) if config.param_names_for_lifetime_elision_hints => {
                    if let Some(c) = used_names.get_mut(it.text().as_str()) {
                        *c += 1;
                        format_smolstr!("'{}{c}", it.text().as_str())
                    } else {
                        used_names.insert(it.text().as_str().into(), 0);
                        format_smolstr!("'{}", it.text().as_str())
                    }
                }
                _ => gen_idx_name(),
            };
            allocated_lifetimes.push(name);
        });
    }

    // fetch output lifetime if elision rule applies
    let output = match potential_lt_refs.as_slice() {
        [(_, _, lifetime, _), ..] if self_param.is_some() || potential_lt_refs.len() == 1 => {
            match lifetime {
                Some(lt) => match lt.text().as_str() {
                    "'_" => allocated_lifetimes.first().cloned(),
                    "'static" => None,
                    name => Some(name.into()),
                },
                None => allocated_lifetimes.first().cloned(),
            }
        }
        [..] => None,
    };

    if allocated_lifetimes.is_empty() && output.is_none() {
        return None;
    }

    // apply hints
    // apply output if required
    if let (Some(output_lt), Some(r)) = (&output, ret_type) {
        if let Some(ty) = r.ty() {
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(ty) if ty.lifetime().is_none() => {
                    if let Some(amp) = ty.amp_token() {
                        is_trivial = false;
                        acc.push(mk_lt_hint(amp, output_lt.to_string()));
                    }
                    false
                }
                ast::Type::FnPtrType(_) => {
                    is_trivial = false;
                    true
                }
                ast::Type::PathType(t) => {
                    if t.path()
                        .and_then(|it| it.segment())
                        .and_then(|it| it.parenthesized_arg_list())
                        .is_some()
                    {
                        is_trivial = false;
                        true
                    } else {
                        false
                    }
                }
                _ => false,
            })
        }
    }

    if config.lifetime_elision_hints == LifetimeElisionHints::SkipTrivial && is_trivial {
        return None;
    }

    let mut a = allocated_lifetimes.iter();
    for (_, amp_token, _, is_elided) in potential_lt_refs {
        if is_elided {
            let t = amp_token?;
            let lt = a.next()?;
            acc.push(mk_lt_hint(t, lt.to_string()));
        }
    }

    // generate generic param list things
    match (generic_param_list, allocated_lifetimes.as_slice()) {
        (_, []) => (),
        (Some(gpl), allocated_lifetimes) => {
            let angle_tok = gpl.l_angle_token()?;
            let is_empty = gpl.generic_params().next().is_none();
            acc.push(InlayHint {
                range: angle_tok.text_range(),
                kind: InlayKind::Lifetime,
                label: format!(
                    "{}{}",
                    allocated_lifetimes.iter().format(", "),
                    if is_empty { "" } else { ", " }
                )
                .into(),
                text_edit: None,
                position: InlayHintPosition::After,
                pad_left: false,
                pad_right: true,
                resolve_parent: None,
            });
        }
        (None, allocated_lifetimes) => on_missing_gpl(acc, allocated_lifetimes),
    }
    if let Some(stack) = ctx.lifetime_stacks.last_mut() {
        stack.extend(allocated_lifetimes);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::{
        InlayHintsConfig, LifetimeElisionHints,
        inlay_hints::tests::{TEST_CONFIG, check, check_with_config},
    };

    #[test]
    fn hints_lifetimes() {
        check(
            r#"
fn empty() {}

fn no_gpl(a: &()) {}
 //^^^^^^<'0>
          // ^'0
fn empty_gpl<>(a: &()) {}
      //    ^'0   ^'0
fn partial<'b>(a: &(), b: &'b ()) {}
//        ^'0, $  ^'0
fn partial<'a>(a: &'a (), b: &()) {}
//        ^'0, $             ^'0

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}
// ^^^^^^^^<'0, '1>
            // ^'0     ^'1

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
// ^^^^^^^^^<'0, '1, '2>
              //^'0 ^'1 ^'2
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    // ^^^<'0>
        // ^'0
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn hints_lifetimes_named() {
        check_with_config(
            InlayHintsConfig { param_names_for_lifetime_elision_hints: true, ..TEST_CONFIG },
            r#"
fn nested_in<'named>(named: &        &X<      &()>) {}
//          ^'named1, 'named2, 'named3, $
                          //^'named1 ^'named2 ^'named3
"#,
        );
    }

    #[test]
    fn hints_lifetimes_trivial_skip() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::SkipTrivial,
                ..TEST_CONFIG
            },
            r#"
fn no_gpl(a: &()) {}
fn empty_gpl<>(a: &()) {}
fn partial<'b>(a: &(), b: &'b ()) {}
fn partial<'a>(a: &'a (), b: &()) {}

fn single_ret(a: &()) -> &() {}
// ^^^^^^^^^^<'0>
              // ^'0     ^'0
fn full_mul(a: &(), b: &()) {}

fn foo<'c>(a: &'c ()) -> &() {}
                      // ^'c

fn nested_in(a: &   &X< &()>) {}
fn nested_out(a: &()) -> &   &X< &()>{}
// ^^^^^^^^^^<'0>
               //^'0     ^'0 ^'0 ^'0

impl () {
    fn foo(&self) {}
    fn foo(&self) -> &() {}
    // ^^^<'0>
        // ^'0       ^'0
    fn foo(&self, a: &()) -> &() {}
    // ^^^<'0, '1>
        // ^'0       ^'1     ^'0
}
"#,
        );
    }

    #[test]
    fn no_collide() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                param_names_for_lifetime_elision_hints: true,
                ..TEST_CONFIG
            },
            r#"
impl<'foo> {
    fn foo(foo: &()) {}
    // ^^^ <'foo1>
             // ^ 'foo1
}
"#,
        );
    }

    #[test]
    fn hints_lifetimes_fn_ptr() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
fn fn_ptr(a: fn(&()) -> &fn(&()) -> &()) {}
           //^^ for<'0>
              //^'0
                      //^'0
                       //^^ for<'1>
                          //^'1
                                  //^'1
fn fn_ptr2(a: for<'a> fn(&()) -> &()) {}
               //^'0, $
                       //^'0
                               //^'0
fn fn_trait(a: &impl Fn(&()) -> &()) {}
// ^^^^^^^^<'0>
            // ^'0
                  // ^^ for<'1>
                      //^'1
                             // ^'1
"#,
        );
    }

    #[test]
    fn hints_in_non_gen_defs() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
const _: fn(&()) -> &();
       //^^ for<'0>
          //^'0
                  //^'0
"#,
        );
    }
}
