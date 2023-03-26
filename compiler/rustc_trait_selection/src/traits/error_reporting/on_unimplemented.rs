use super::{ObligationCauseCode, PredicateObligation};
use crate::infer::error_reporting::TypeErrCtxt;
use rustc_ast::{MetaItem, NestedMetaItem};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{struct_span_err, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::SubstsRef;
use rustc_middle::ty::{self, GenericParamDefKind, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::{Span, DUMMY_SP};
use std::iter;

use crate::errors::{
    EmptyOnClauseInOnUnimplemented, InvalidOnClauseInOnUnimplemented, NoValueInOnUnimplemented,
};

use super::InferCtxtPrivExt;

pub trait TypeErrCtxtExt<'tcx> {
    /*private*/
    fn impl_similar_to(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<(DefId, SubstsRef<'tcx>)>;

    /*private*/
    fn describe_enclosure(&self, hir_id: hir::HirId) -> Option<&'static str>;

    fn on_unimplemented_note(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> OnUnimplementedNote;
}

/// The symbols which are always allowed in a format string
static ALLOWED_FORMAT_SYMBOLS: &[Symbol] = &[
    kw::SelfUpper,
    sym::ItemContext,
    sym::from_method,
    sym::from_desugaring,
    sym::direct,
    sym::cause,
    sym::integral,
    sym::integer_,
    sym::float,
    sym::_Self,
    sym::crate_local,
];

impl<'tcx> TypeErrCtxtExt<'tcx> for TypeErrCtxt<'_, 'tcx> {
    fn impl_similar_to(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> Option<(DefId, SubstsRef<'tcx>)> {
        let tcx = self.tcx;
        let param_env = obligation.param_env;
        let trait_ref = self.instantiate_binder_with_placeholders(trait_ref);
        let trait_self_ty = trait_ref.self_ty();

        let mut self_match_impls = vec![];
        let mut fuzzy_match_impls = vec![];

        self.tcx.for_each_relevant_impl(trait_ref.def_id, trait_self_ty, |def_id| {
            let impl_substs = self.fresh_substs_for_item(obligation.cause.span, def_id);
            let impl_trait_ref = tcx.impl_trait_ref(def_id).unwrap().subst(tcx, impl_substs);

            let impl_self_ty = impl_trait_ref.self_ty();

            if self.can_eq(param_env, trait_self_ty, impl_self_ty) {
                self_match_impls.push((def_id, impl_substs));

                if iter::zip(
                    trait_ref.substs.types().skip(1),
                    impl_trait_ref.substs.types().skip(1),
                )
                .all(|(u, v)| self.fuzzy_match_tys(u, v, false).is_some())
                {
                    fuzzy_match_impls.push((def_id, impl_substs));
                }
            }
        });

        let impl_def_id_and_substs = if self_match_impls.len() == 1 {
            self_match_impls[0]
        } else if fuzzy_match_impls.len() == 1 {
            fuzzy_match_impls[0]
        } else {
            return None;
        };

        tcx.has_attr(impl_def_id_and_substs.0, sym::rustc_on_unimplemented)
            .then_some(impl_def_id_and_substs)
    }

    /// Used to set on_unimplemented's `ItemContext`
    /// to be the enclosing (async) block/function/closure
    fn describe_enclosure(&self, hir_id: hir::HirId) -> Option<&'static str> {
        let hir = self.tcx.hir();
        let node = hir.find(hir_id)?;
        match &node {
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Fn(sig, _, body_id), .. }) => {
                self.describe_generator(*body_id).or_else(|| {
                    Some(match sig.header {
                        hir::FnHeader { asyncness: hir::IsAsync::Async, .. } => "an async function",
                        _ => "a function",
                    })
                })
            }
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(_, hir::TraitFn::Provided(body_id)),
                ..
            }) => self.describe_generator(*body_id).or_else(|| Some("a trait method")),
            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, body_id),
                ..
            }) => self.describe_generator(*body_id).or_else(|| {
                Some(match sig.header {
                    hir::FnHeader { asyncness: hir::IsAsync::Async, .. } => "an async method",
                    _ => "a method",
                })
            }),
            hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Closure(hir::Closure { body, movability, .. }),
                ..
            }) => self.describe_generator(*body).or_else(|| {
                Some(if movability.is_some() { "an async closure" } else { "a closure" })
            }),
            hir::Node::Expr(hir::Expr { .. }) => {
                let parent_hid = hir.parent_id(hir_id);
                if parent_hid != hir_id { self.describe_enclosure(parent_hid) } else { None }
            }
            _ => None,
        }
    }

    fn on_unimplemented_note(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        obligation: &PredicateObligation<'tcx>,
    ) -> OnUnimplementedNote {
        let (def_id, substs) = self
            .impl_similar_to(trait_ref, obligation)
            .unwrap_or_else(|| (trait_ref.def_id(), trait_ref.skip_binder().substs));
        let trait_ref = trait_ref.skip_binder();

        let mut flags = vec![];
        // FIXME(-Zlower-impl-trait-in-trait-to-assoc-ty): HIR is not present for RPITITs,
        // but I guess we could synthesize one here. We don't see any errors that rely on
        // that yet, though.
        let enclosure =
            if let Some(body_hir) = self.tcx.opt_local_def_id_to_hir_id(obligation.cause.body_id) {
                self.describe_enclosure(body_hir).map(|s| s.to_owned())
            } else {
                None
            };
        flags.push((sym::ItemContext, enclosure));

        match obligation.cause.code() {
            ObligationCauseCode::BuiltinDerivedObligation(..)
            | ObligationCauseCode::ImplDerivedObligation(..)
            | ObligationCauseCode::DerivedObligation(..) => {}
            _ => {
                // this is a "direct", user-specified, rather than derived,
                // obligation.
                flags.push((sym::direct, None));
            }
        }

        if let ObligationCauseCode::ItemObligation(item)
        | ObligationCauseCode::BindingObligation(item, _)
        | ObligationCauseCode::ExprItemObligation(item, ..)
        | ObligationCauseCode::ExprBindingObligation(item, ..) = *obligation.cause.code()
        {
            // FIXME: maybe also have some way of handling methods
            // from other traits? That would require name resolution,
            // which we might want to be some sort of hygienic.
            //
            // Currently I'm leaving it for what I need for `try`.
            if self.tcx.trait_of_item(item) == Some(trait_ref.def_id) {
                let method = self.tcx.item_name(item);
                flags.push((sym::from_method, None));
                flags.push((sym::from_method, Some(method.to_string())));
            }
        }

        if let Some(k) = obligation.cause.span.desugaring_kind() {
            flags.push((sym::from_desugaring, None));
            flags.push((sym::from_desugaring, Some(format!("{:?}", k))));
        }

        if let ObligationCauseCode::MainFunctionType = obligation.cause.code() {
            flags.push((sym::cause, Some("MainFunctionType".to_string())));
        }

        // Add all types without trimmed paths.
        ty::print::with_no_trimmed_paths!({
            let generics = self.tcx.generics_of(def_id);
            let self_ty = trait_ref.self_ty();
            // This is also included through the generics list as `Self`,
            // but the parser won't allow you to use it
            flags.push((sym::_Self, Some(self_ty.to_string())));
            if let Some(def) = self_ty.ty_adt_def() {
                // We also want to be able to select self's original
                // signature with no type arguments resolved
                flags.push((
                    sym::_Self,
                    Some(self.tcx.type_of(def.did()).subst_identity().to_string()),
                ));
            }

            for param in generics.params.iter() {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        substs[param.index as usize].to_string()
                    }
                    GenericParamDefKind::Lifetime => continue,
                };
                let name = param.name;
                flags.push((name, Some(value)));

                if let GenericParamDefKind::Type { .. } = param.kind {
                    let param_ty = substs[param.index as usize].expect_ty();
                    if let Some(def) = param_ty.ty_adt_def() {
                        // We also want to be able to select the parameter's
                        // original signature with no type arguments resolved
                        flags.push((
                            name,
                            Some(self.tcx.type_of(def.did()).subst_identity().to_string()),
                        ));
                    }
                }
            }

            if let Some(true) = self_ty.ty_adt_def().map(|def| def.did().is_local()) {
                flags.push((sym::crate_local, None));
            }

            // Allow targeting all integers using `{integral}`, even if the exact type was resolved
            if self_ty.is_integral() {
                flags.push((sym::_Self, Some("{integral}".to_owned())));
            }

            if self_ty.is_array_slice() {
                flags.push((sym::_Self, Some("&[]".to_owned())));
            }

            if self_ty.is_fn() {
                let fn_sig = self_ty.fn_sig(self.tcx);
                let shortname = match fn_sig.unsafety() {
                    hir::Unsafety::Normal => "fn",
                    hir::Unsafety::Unsafe => "unsafe fn",
                };
                flags.push((sym::_Self, Some(shortname.to_owned())));
            }

            // Slices give us `[]`, `[{ty}]`
            if let ty::Slice(aty) = self_ty.kind() {
                flags.push((sym::_Self, Some("[]".to_string())));
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the slice's type's original
                    // signature with no type arguments resolved
                    flags.push((
                        sym::_Self,
                        Some(format!("[{}]", self.tcx.type_of(def.did()).subst_identity())),
                    ));
                }
                if aty.is_integral() {
                    flags.push((sym::_Self, Some("[{integral}]".to_string())));
                }
            }

            // Arrays give us `[]`, `[{ty}; _]` and `[{ty}; N]`
            if let ty::Array(aty, len) = self_ty.kind() {
                flags.push((sym::_Self, Some("[]".to_string())));
                let len = len.kind().try_to_value().and_then(|v| v.try_to_target_usize(self.tcx));
                flags.push((sym::_Self, Some(format!("[{}; _]", aty))));
                if let Some(n) = len {
                    flags.push((sym::_Self, Some(format!("[{}; {}]", aty, n))));
                }
                if let Some(def) = aty.ty_adt_def() {
                    // We also want to be able to select the array's type's original
                    // signature with no type arguments resolved
                    let def_ty = self.tcx.type_of(def.did()).subst_identity();
                    flags.push((sym::_Self, Some(format!("[{def_ty}; _]"))));
                    if let Some(n) = len {
                        flags.push((sym::_Self, Some(format!("[{def_ty}; {n}]"))));
                    }
                }
                if aty.is_integral() {
                    flags.push((sym::_Self, Some("[{integral}; _]".to_string())));
                    if let Some(n) = len {
                        flags.push((sym::_Self, Some(format!("[{{integral}}; {n}]"))));
                    }
                }
            }
            if let ty::Dynamic(traits, _, _) = self_ty.kind() {
                for t in traits.iter() {
                    if let ty::ExistentialPredicate::Trait(trait_ref) = t.skip_binder() {
                        flags.push((sym::_Self, Some(self.tcx.def_path_str(trait_ref.def_id))))
                    }
                }
            }
        });

        if let Ok(Some(command)) = OnUnimplementedDirective::of_item(self.tcx, def_id) {
            command.evaluate(self.tcx, trait_ref, &flags)
        } else {
            OnUnimplementedNote::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnUnimplementedFormatString(Symbol);

#[derive(Debug)]
pub struct OnUnimplementedDirective {
    pub condition: Option<MetaItem>,
    pub subcommands: Vec<OnUnimplementedDirective>,
    pub message: Option<OnUnimplementedFormatString>,
    pub label: Option<OnUnimplementedFormatString>,
    pub note: Option<OnUnimplementedFormatString>,
    pub parent_label: Option<OnUnimplementedFormatString>,
    pub append_const_msg: Option<Option<Symbol>>,
}

/// For the `#[rustc_on_unimplemented]` attribute
#[derive(Default)]
pub struct OnUnimplementedNote {
    pub message: Option<String>,
    pub label: Option<String>,
    pub note: Option<String>,
    pub parent_label: Option<String>,
    /// Append a message for `~const Trait` errors. `None` means not requested and
    /// should fallback to a generic message, `Some(None)` suggests using the default
    /// appended message, `Some(Some(s))` suggests use the `s` message instead of the
    /// default one..
    pub append_const_msg: Option<Option<Symbol>>,
}

impl<'tcx> OnUnimplementedDirective {
    fn parse(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        items: &[NestedMetaItem],
        span: Span,
        is_root: bool,
    ) -> Result<Self, ErrorGuaranteed> {
        let mut errored = None;
        let mut item_iter = items.iter();

        let parse_value = |value_str| {
            OnUnimplementedFormatString::try_parse(tcx, item_def_id, value_str, span).map(Some)
        };

        let condition = if is_root {
            None
        } else {
            let cond = item_iter
                .next()
                .ok_or_else(|| tcx.sess.emit_err(EmptyOnClauseInOnUnimplemented { span }))?
                .meta_item()
                .ok_or_else(|| tcx.sess.emit_err(InvalidOnClauseInOnUnimplemented { span }))?;
            attr::eval_condition(cond, &tcx.sess.parse_sess, Some(tcx.features()), &mut |cfg| {
                if let Some(value) = cfg.value && let Err(guar) = parse_value(value) {
                    errored = Some(guar);
                }
                true
            });
            Some(cond.clone())
        };

        let mut message = None;
        let mut label = None;
        let mut note = None;
        let mut parent_label = None;
        let mut subcommands = vec![];
        let mut append_const_msg = None;

        for item in item_iter {
            if item.has_name(sym::message) && message.is_none() {
                if let Some(message_) = item.value_str() {
                    message = parse_value(message_)?;
                    continue;
                }
            } else if item.has_name(sym::label) && label.is_none() {
                if let Some(label_) = item.value_str() {
                    label = parse_value(label_)?;
                    continue;
                }
            } else if item.has_name(sym::note) && note.is_none() {
                if let Some(note_) = item.value_str() {
                    note = parse_value(note_)?;
                    continue;
                }
            } else if item.has_name(sym::parent_label) && parent_label.is_none() {
                if let Some(parent_label_) = item.value_str() {
                    parent_label = parse_value(parent_label_)?;
                    continue;
                }
            } else if item.has_name(sym::on)
                && is_root
                && message.is_none()
                && label.is_none()
                && note.is_none()
            {
                if let Some(items) = item.meta_item_list() {
                    match Self::parse(tcx, item_def_id, &items, item.span(), false) {
                        Ok(subcommand) => subcommands.push(subcommand),
                        Err(reported) => errored = Some(reported),
                    };
                    continue;
                }
            } else if item.has_name(sym::append_const_msg) && append_const_msg.is_none() {
                if let Some(msg) = item.value_str() {
                    append_const_msg = Some(Some(msg));
                    continue;
                } else if item.is_word() {
                    append_const_msg = Some(None);
                    continue;
                }
            }

            // nothing found
            tcx.sess.emit_err(NoValueInOnUnimplemented { span: item.span() });
        }

        if let Some(reported) = errored {
            Err(reported)
        } else {
            Ok(OnUnimplementedDirective {
                condition,
                subcommands,
                message,
                label,
                note,
                parent_label,
                append_const_msg,
            })
        }
    }

    pub fn of_item(tcx: TyCtxt<'tcx>, item_def_id: DefId) -> Result<Option<Self>, ErrorGuaranteed> {
        let Some(attr) = tcx.get_attr(item_def_id, sym::rustc_on_unimplemented) else {
            return Ok(None);
        };

        let result = if let Some(items) = attr.meta_item_list() {
            Self::parse(tcx, item_def_id, &items, attr.span, true).map(Some)
        } else if let Some(value) = attr.value_str() {
            Ok(Some(OnUnimplementedDirective {
                condition: None,
                message: None,
                subcommands: vec![],
                label: Some(OnUnimplementedFormatString::try_parse(
                    tcx,
                    item_def_id,
                    value,
                    attr.span,
                )?),
                note: None,
                parent_label: None,
                append_const_msg: None,
            }))
        } else {
            let reported =
                tcx.sess.delay_span_bug(DUMMY_SP, "of_item: neither meta_item_list nor value_str");
            return Err(reported);
        };
        debug!("of_item({:?}) = {:?}", item_def_id, result);
        result
    }

    pub fn evaluate(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        options: &[(Symbol, Option<String>)],
    ) -> OnUnimplementedNote {
        let mut message = None;
        let mut label = None;
        let mut note = None;
        let mut parent_label = None;
        let mut append_const_msg = None;
        info!("evaluate({:?}, trait_ref={:?}, options={:?})", self, trait_ref, options);

        let options_map: FxHashMap<Symbol, String> =
            options.iter().filter_map(|(k, v)| v.clone().map(|v| (*k, v))).collect();

        for command in self.subcommands.iter().chain(Some(self)).rev() {
            if let Some(ref condition) = command.condition && !attr::eval_condition(
                condition,
                &tcx.sess.parse_sess,
                Some(tcx.features()),
                &mut |cfg| {
                    let value = cfg.value.map(|v| {
                        OnUnimplementedFormatString(v).format(tcx, trait_ref, &options_map)
                    });

                    options.contains(&(cfg.name, value))
                },
            ) {
                debug!("evaluate: skipping {:?} due to condition", command);
                continue;
            }
            debug!("evaluate: {:?} succeeded", command);
            if let Some(ref message_) = command.message {
                message = Some(message_.clone());
            }

            if let Some(ref label_) = command.label {
                label = Some(label_.clone());
            }

            if let Some(ref note_) = command.note {
                note = Some(note_.clone());
            }

            if let Some(ref parent_label_) = command.parent_label {
                parent_label = Some(parent_label_.clone());
            }

            append_const_msg = command.append_const_msg;
        }

        OnUnimplementedNote {
            label: label.map(|l| l.format(tcx, trait_ref, &options_map)),
            message: message.map(|m| m.format(tcx, trait_ref, &options_map)),
            note: note.map(|n| n.format(tcx, trait_ref, &options_map)),
            parent_label: parent_label.map(|e_s| e_s.format(tcx, trait_ref, &options_map)),
            append_const_msg,
        }
    }
}

impl<'tcx> OnUnimplementedFormatString {
    fn try_parse(
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        from: Symbol,
        err_sp: Span,
    ) -> Result<Self, ErrorGuaranteed> {
        let result = OnUnimplementedFormatString(from);
        result.verify(tcx, item_def_id, err_sp)?;
        Ok(result)
    }

    fn verify(
        &self,
        tcx: TyCtxt<'tcx>,
        item_def_id: DefId,
        span: Span,
    ) -> Result<(), ErrorGuaranteed> {
        let trait_def_id = if tcx.is_trait(item_def_id) {
            item_def_id
        } else {
            tcx.trait_id_of_impl(item_def_id)
                .expect("expected `on_unimplemented` to correspond to a trait")
        };
        let trait_name = tcx.item_name(trait_def_id);
        let generics = tcx.generics_of(item_def_id);
        let s = self.0.as_str();
        let parser = Parser::new(s, None, None, false, ParseMode::Format);
        let mut result = Ok(());
        for token in parser {
            match token {
                Piece::String(_) => (), // Normal string, no need to check it
                Piece::NextArgument(a) => match a.position {
                    Position::ArgumentNamed(s) => {
                        match Symbol::intern(s) {
                            // `{ThisTraitsName}` is allowed
                            s if s == trait_name => (),
                            s if ALLOWED_FORMAT_SYMBOLS.contains(&s) => (),
                            // So is `{A}` if A is a type parameter
                            s if generics.params.iter().any(|param| param.name == s) => (),
                            s => {
                                result = Err(struct_span_err!(
                                    tcx.sess,
                                    span,
                                    E0230,
                                    "there is no parameter `{}` on {}",
                                    s,
                                    if trait_def_id == item_def_id {
                                        format!("trait `{}`", trait_name)
                                    } else {
                                        "impl".to_string()
                                    }
                                )
                                .emit());
                            }
                        }
                    }
                    // `{:1}` and `{}` are not to be used
                    Position::ArgumentIs(..) | Position::ArgumentImplicitlyIs(_) => {
                        let reported = struct_span_err!(
                            tcx.sess,
                            span,
                            E0231,
                            "only named substitution parameters are allowed"
                        )
                        .emit();
                        result = Err(reported);
                    }
                },
            }
        }

        result
    }

    pub fn format(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
        options: &FxHashMap<Symbol, String>,
    ) -> String {
        let name = tcx.item_name(trait_ref.def_id);
        let trait_str = tcx.def_path_str(trait_ref.def_id);
        let generics = tcx.generics_of(trait_ref.def_id);
        let generic_map = generics
            .params
            .iter()
            .filter_map(|param| {
                let value = match param.kind {
                    GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. } => {
                        trait_ref.substs[param.index as usize].to_string()
                    }
                    GenericParamDefKind::Lifetime => return None,
                };
                let name = param.name;
                Some((name, value))
            })
            .collect::<FxHashMap<Symbol, String>>();
        let empty_string = String::new();

        let s = self.0.as_str();
        let parser = Parser::new(s, None, None, false, ParseMode::Format);
        let item_context = (options.get(&sym::ItemContext)).unwrap_or(&empty_string);
        parser
            .map(|p| match p {
                Piece::String(s) => s,
                Piece::NextArgument(a) => match a.position {
                    Position::ArgumentNamed(s) => {
                        let s = Symbol::intern(s);
                        match generic_map.get(&s) {
                            Some(val) => val,
                            None if s == name => &trait_str,
                            None => {
                                if let Some(val) = options.get(&s) {
                                    val
                                } else if s == sym::from_desugaring || s == sym::from_method {
                                    // don't break messages using these two arguments incorrectly
                                    &empty_string
                                } else if s == sym::ItemContext {
                                    &item_context
                                } else if s == sym::integral {
                                    "{integral}"
                                } else if s == sym::integer_ {
                                    "{integer}"
                                } else if s == sym::float {
                                    "{float}"
                                } else {
                                    bug!(
                                        "broken on_unimplemented {:?} for {:?}: \
                                      no argument matching {:?}",
                                        self.0,
                                        trait_ref,
                                        s
                                    )
                                }
                            }
                        }
                    }
                    _ => bug!("broken on_unimplemented {:?} - bad format arg", self.0),
                },
            })
            .collect()
    }
}
