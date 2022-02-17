use rustc_ast::{MetaItem, NestedMetaItem};
use rustc_attr as attr;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{struct_span_err, ErrorReported};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, GenericParamDefKind, TyCtxt};
use rustc_parse_format::{ParseMode, Parser, Piece, Position};
use rustc_span::symbol::{kw, sym, Symbol};
use rustc_span::Span;

#[derive(Clone, Debug)]
pub struct OnUnimplementedFormatString(Symbol);

#[derive(Debug)]
pub struct OnUnimplementedDirective {
    pub condition: Option<MetaItem>,
    pub subcommands: Vec<OnUnimplementedDirective>,
    pub message: Option<OnUnimplementedFormatString>,
    pub label: Option<OnUnimplementedFormatString>,
    pub note: Option<OnUnimplementedFormatString>,
    pub enclosing_scope: Option<OnUnimplementedFormatString>,
    pub append_const_msg: Option<Option<Symbol>>,
}

#[derive(Default)]
pub struct OnUnimplementedNote {
    pub message: Option<String>,
    pub label: Option<String>,
    pub note: Option<String>,
    pub enclosing_scope: Option<String>,
    /// Append a message for `~const Trait` errors. `None` means not requested and
    /// should fallback to a generic message, `Some(None)` suggests using the default
    /// appended message, `Some(Some(s))` suggests use the `s` message instead of the
    /// default one..
    pub append_const_msg: Option<Option<Symbol>>,
}

fn parse_error(
    tcx: TyCtxt<'_>,
    span: Span,
    message: &str,
    label: &str,
    note: Option<&str>,
) -> ErrorReported {
    let mut diag = struct_span_err!(tcx.sess, span, E0232, "{}", message);
    diag.span_label(span, label);
    if let Some(note) = note {
        diag.note(note);
    }
    diag.emit();
    ErrorReported
}

impl<'tcx> OnUnimplementedDirective {
    fn parse(
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        items: &[NestedMetaItem],
        span: Span,
        is_root: bool,
    ) -> Result<Self, ErrorReported> {
        let mut errored = false;
        let mut item_iter = items.iter();

        let parse_value = |value_str| {
            OnUnimplementedFormatString::try_parse(tcx, trait_def_id, value_str, span).map(Some)
        };

        let condition = if is_root {
            None
        } else {
            let cond = item_iter
                .next()
                .ok_or_else(|| {
                    parse_error(
                        tcx,
                        span,
                        "empty `on`-clause in `#[rustc_on_unimplemented]`",
                        "empty on-clause here",
                        None,
                    )
                })?
                .meta_item()
                .ok_or_else(|| {
                    parse_error(
                        tcx,
                        span,
                        "invalid `on`-clause in `#[rustc_on_unimplemented]`",
                        "invalid on-clause here",
                        None,
                    )
                })?;
            attr::eval_condition(cond, &tcx.sess.parse_sess, Some(tcx.features()), &mut |item| {
                if let Some(symbol) = item.value_str() {
                    if parse_value(symbol).is_err() {
                        errored = true;
                    }
                }
                true
            });
            Some(cond.clone())
        };

        let mut message = None;
        let mut label = None;
        let mut note = None;
        let mut enclosing_scope = None;
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
            } else if item.has_name(sym::enclosing_scope) && enclosing_scope.is_none() {
                if let Some(enclosing_scope_) = item.value_str() {
                    enclosing_scope = parse_value(enclosing_scope_)?;
                    continue;
                }
            } else if item.has_name(sym::on)
                && is_root
                && message.is_none()
                && label.is_none()
                && note.is_none()
            {
                if let Some(items) = item.meta_item_list() {
                    if let Ok(subcommand) =
                        Self::parse(tcx, trait_def_id, &items, item.span(), false)
                    {
                        subcommands.push(subcommand);
                    } else {
                        errored = true;
                    }
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
            parse_error(
                tcx,
                item.span(),
                "this attribute must have a valid value",
                "expected value here",
                Some(r#"eg `#[rustc_on_unimplemented(message="foo")]`"#),
            );
        }

        if errored {
            Err(ErrorReported)
        } else {
            Ok(OnUnimplementedDirective {
                condition,
                subcommands,
                message,
                label,
                note,
                enclosing_scope,
                append_const_msg,
            })
        }
    }

    pub fn of_item(
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        impl_def_id: DefId,
    ) -> Result<Option<Self>, ErrorReported> {
        let attrs = tcx.get_attrs(impl_def_id);

        let Some(attr) = tcx.sess.find_by_name(&attrs, sym::rustc_on_unimplemented) else {
            return Ok(None);
        };

        let result = if let Some(items) = attr.meta_item_list() {
            Self::parse(tcx, trait_def_id, &items, attr.span, true).map(Some)
        } else if let Some(value) = attr.value_str() {
            Ok(Some(OnUnimplementedDirective {
                condition: None,
                message: None,
                subcommands: vec![],
                label: Some(OnUnimplementedFormatString::try_parse(
                    tcx,
                    trait_def_id,
                    value,
                    attr.span,
                )?),
                note: None,
                enclosing_scope: None,
                append_const_msg: None,
            }))
        } else {
            return Err(ErrorReported);
        };
        debug!("of_item({:?}/{:?}) = {:?}", trait_def_id, impl_def_id, result);
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
        let mut enclosing_scope = None;
        let mut append_const_msg = None;
        info!("evaluate({:?}, trait_ref={:?}, options={:?})", self, trait_ref, options);

        let options_map: FxHashMap<Symbol, String> =
            options.iter().filter_map(|(k, v)| v.as_ref().map(|v| (*k, v.to_owned()))).collect();

        for command in self.subcommands.iter().chain(Some(self)).rev() {
            if let Some(ref condition) = command.condition {
                if !attr::eval_condition(
                    condition,
                    &tcx.sess.parse_sess,
                    Some(tcx.features()),
                    &mut |c| {
                        c.ident().map_or(false, |ident| {
                            let value = c.value_str().map(|s| {
                                OnUnimplementedFormatString(s).format(tcx, trait_ref, &options_map)
                            });

                            options.contains(&(ident.name, value))
                        })
                    },
                ) {
                    debug!("evaluate: skipping {:?} due to condition", command);
                    continue;
                }
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

            if let Some(ref enclosing_scope_) = command.enclosing_scope {
                enclosing_scope = Some(enclosing_scope_.clone());
            }

            append_const_msg = command.append_const_msg.clone();
        }

        OnUnimplementedNote {
            label: label.map(|l| l.format(tcx, trait_ref, &options_map)),
            message: message.map(|m| m.format(tcx, trait_ref, &options_map)),
            note: note.map(|n| n.format(tcx, trait_ref, &options_map)),
            enclosing_scope: enclosing_scope.map(|e_s| e_s.format(tcx, trait_ref, &options_map)),
            append_const_msg,
        }
    }
}

impl<'tcx> OnUnimplementedFormatString {
    fn try_parse(
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        from: Symbol,
        err_sp: Span,
    ) -> Result<Self, ErrorReported> {
        let result = OnUnimplementedFormatString(from);
        result.verify(tcx, trait_def_id, err_sp)?;
        Ok(result)
    }

    fn verify(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
        span: Span,
    ) -> Result<(), ErrorReported> {
        let name = tcx.item_name(trait_def_id);
        let generics = tcx.generics_of(trait_def_id);
        let s = self.0.as_str();
        let parser = Parser::new(s, None, None, false, ParseMode::Format);
        let mut result = Ok(());
        for token in parser {
            match token {
                Piece::String(_) => (), // Normal string, no need to check it
                Piece::NextArgument(a) => match a.position {
                    // `{Self}` is allowed
                    Position::ArgumentNamed(s, _) if s == kw::SelfUpper => (),
                    // `{ThisTraitsName}` is allowed
                    Position::ArgumentNamed(s, _) if s == name => (),
                    // `{from_method}` is allowed
                    Position::ArgumentNamed(s, _) if s == sym::from_method => (),
                    // `{from_desugaring}` is allowed
                    Position::ArgumentNamed(s, _) if s == sym::from_desugaring => (),
                    // `{ItemContext}` is allowed
                    Position::ArgumentNamed(s, _) if s == sym::ItemContext => (),
                    // `{integral}` and `{integer}` and `{float}` are allowed
                    Position::ArgumentNamed(s, _)
                        if s == sym::integral || s == sym::integer_ || s == sym::float =>
                    {
                        ()
                    }
                    // So is `{A}` if A is a type parameter
                    Position::ArgumentNamed(s, _) => {
                        match generics.params.iter().find(|param| param.name == s) {
                            Some(_) => (),
                            None => {
                                struct_span_err!(
                                    tcx.sess,
                                    span,
                                    E0230,
                                    "there is no parameter `{}` on trait `{}`",
                                    s,
                                    name
                                )
                                .emit();
                                result = Err(ErrorReported);
                            }
                        }
                    }
                    // `{:1}` and `{}` are not to be used
                    Position::ArgumentIs(_) | Position::ArgumentImplicitlyIs(_) => {
                        struct_span_err!(
                            tcx.sess,
                            span,
                            E0231,
                            "only named substitution parameters are allowed"
                        )
                        .emit();
                        result = Err(ErrorReported);
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
                    Position::ArgumentNamed(s, _) => match generic_map.get(&s) {
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
                    },
                    _ => bug!("broken on_unimplemented {:?} - bad format arg", self.0),
                },
            })
            .collect()
    }
}
