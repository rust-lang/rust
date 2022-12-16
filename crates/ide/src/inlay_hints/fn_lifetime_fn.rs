use ide_db::{syntax_helpers::node_ext::walk_ty, FxHashMap};
use itertools::Itertools;
use syntax::SmolStr;
use syntax::{
    ast::{self, AstNode, HasGenericParams, HasName},
    SyntaxToken,
};

use crate::{InlayHint, InlayHintsConfig, InlayKind, InlayTooltip, LifetimeElisionHints};

pub(super) fn hints(
    acc: &mut Vec<InlayHint>,
    config: &InlayHintsConfig,
    func: ast::Fn,
) -> Option<()> {
    if config.lifetime_elision_hints == LifetimeElisionHints::Never {
        return None;
    }

    let mk_lt_hint = |t: SyntaxToken, label: String| InlayHint {
        range: t.text_range(),
        kind: InlayKind::LifetimeHint,
        label: label.into(),
        tooltip: Some(InlayTooltip::String("Elided lifetime".into())),
    };

    let param_list = func.param_list()?;
    let generic_param_list = func.generic_param_list();
    let ret_type = func.ret_type();
    let self_param = param_list.self_param().filter(|it| it.amp_token().is_some());

    let is_elided = |lt: &Option<ast::Lifetime>| match lt {
        Some(lt) => matches!(lt.text().as_str(), "'_"),
        None => true,
    };

    let potential_lt_refs = {
        let mut acc: Vec<_> = vec![];
        if let Some(self_param) = &self_param {
            let lifetime = self_param.lifetime();
            let is_elided = is_elided(&lifetime);
            acc.push((None, self_param.amp_token(), lifetime, is_elided));
        }
        param_list.params().filter_map(|it| Some((it.pat(), it.ty()?))).for_each(|(pat, ty)| {
            // FIXME: check path types
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(r) => {
                    let lifetime = r.lifetime();
                    let is_elided = is_elided(&lifetime);
                    acc.push((
                        pat.as_ref().and_then(|it| match it {
                            ast::Pat::IdentPat(p) => p.name(),
                            _ => None,
                        }),
                        r.amp_token(),
                        lifetime,
                        is_elided,
                    ))
                }
                _ => (),
            })
        });
        acc
    };

    // allocate names
    let mut gen_idx_name = {
        let mut gen = (0u8..).map(|idx| match idx {
            idx if idx < 10 => SmolStr::from_iter(['\'', (idx + 48) as char]),
            idx => format!("'{idx}").into(),
        });
        move || gen.next().unwrap_or_default()
    };
    let mut allocated_lifetimes = vec![];

    let mut used_names: FxHashMap<SmolStr, usize> =
        match config.param_names_for_lifetime_elision_hints {
            true => generic_param_list
                .iter()
                .flat_map(|gpl| gpl.lifetime_params())
                .filter_map(|param| param.lifetime())
                .filter_map(|lt| Some((SmolStr::from(lt.text().as_str().get(1..)?), 0)))
                .collect(),
            false => Default::default(),
        };
    {
        let mut potential_lt_refs = potential_lt_refs.iter().filter(|&&(.., is_elided)| is_elided);
        if let Some(_) = &self_param {
            if let Some(_) = potential_lt_refs.next() {
                allocated_lifetimes.push(if config.param_names_for_lifetime_elision_hints {
                    // self can't be used as a lifetime, so no need to check for collisions
                    "'self".into()
                } else {
                    gen_idx_name()
                });
            }
        }
        potential_lt_refs.for_each(|(name, ..)| {
            let name = match name {
                Some(it) if config.param_names_for_lifetime_elision_hints => {
                    if let Some(c) = used_names.get_mut(it.text().as_str()) {
                        *c += 1;
                        SmolStr::from(format!("'{text}{c}", text = it.text().as_str()))
                    } else {
                        used_names.insert(it.text().as_str().into(), 0);
                        SmolStr::from_iter(["\'", it.text().as_str()])
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
                    "'_" => allocated_lifetimes.get(0).cloned(),
                    "'static" => None,
                    name => Some(name.into()),
                },
                None => allocated_lifetimes.get(0).cloned(),
            }
        }
        [..] => None,
    };

    if allocated_lifetimes.is_empty() && output.is_none() {
        return None;
    }

    // apply hints
    // apply output if required
    let mut is_trivial = true;
    if let (Some(output_lt), Some(r)) = (&output, ret_type) {
        if let Some(ty) = r.ty() {
            walk_ty(&ty, &mut |ty| match ty {
                ast::Type::RefType(ty) if ty.lifetime().is_none() => {
                    if let Some(amp) = ty.amp_token() {
                        is_trivial = false;
                        acc.push(mk_lt_hint(amp, output_lt.to_string()));
                    }
                }
                _ => (),
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
                kind: InlayKind::LifetimeHint,
                label: format!(
                    "{}{}",
                    allocated_lifetimes.iter().format(", "),
                    if is_empty { "" } else { ", " }
                )
                .into(),
                tooltip: Some(InlayTooltip::String("Elided lifetimes".into())),
            });
        }
        (None, allocated_lifetimes) => acc.push(InlayHint {
            range: func.name()?.syntax().text_range(),
            kind: InlayKind::GenericParamListHint,
            label: format!("<{}>", allocated_lifetimes.iter().format(", "),).into(),
            tooltip: Some(InlayTooltip::String("Elided lifetimes".into())),
        }),
    }
    Some(())
}
