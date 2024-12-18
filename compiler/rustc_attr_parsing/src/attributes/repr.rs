//! Parsing and validation of builtin attributes

use rustc_abi::Align;
use rustc_ast::attr::AttributeExt;
use rustc_ast::{self as ast, MetaItemKind};
use rustc_attr_data_structures::IntType;
use rustc_attr_data_structures::ReprAttr::*;
use rustc_session::Session;
use rustc_span::{Symbol, sym};

use crate::ReprAttr;
use crate::session_diagnostics::{self, IncorrectReprFormatGenericCause};

/// Parse #[repr(...)] forms.
///
/// Valid repr contents: any of the primitive integral type names (see
/// `int_type_of_word`, below) to specify enum discriminant type; `C`, to use
/// the same discriminant size that the corresponding C enum would or C
/// structure layout, `packed` to remove padding, and `transparent` to delegate representation
/// concerns to the only non-ZST field.
pub fn find_repr_attrs(sess: &Session, attr: &impl AttributeExt) -> Vec<ReprAttr> {
    if attr.has_name(sym::repr) { parse_repr_attr(sess, attr) } else { Vec::new() }
}

pub fn parse_repr_attr(sess: &Session, attr: &impl AttributeExt) -> Vec<ReprAttr> {
    assert!(attr.has_name(sym::repr), "expected `#[repr(..)]`, found: {attr:?}");
    let mut acc = Vec::new();
    let dcx = sess.dcx();

    if let Some(items) = attr.meta_item_list() {
        for item in items {
            let mut recognised = false;
            if item.is_word() {
                let hint = match item.name_or_empty() {
                    sym::Rust => Some(ReprRust),
                    sym::C => Some(ReprC),
                    sym::packed => Some(ReprPacked(Align::ONE)),
                    sym::simd => Some(ReprSimd),
                    sym::transparent => Some(ReprTransparent),
                    sym::align => {
                        sess.dcx().emit_err(session_diagnostics::InvalidReprAlignNeedArg {
                            span: item.span(),
                        });
                        recognised = true;
                        None
                    }
                    name => int_type_of_word(name).map(ReprInt),
                };

                if let Some(h) = hint {
                    recognised = true;
                    acc.push(h);
                }
            } else if let Some((name, value)) = item.singleton_lit_list() {
                let mut literal_error = None;
                let mut err_span = item.span();
                if name == sym::align {
                    recognised = true;
                    match parse_alignment(&value.kind) {
                        Ok(literal) => acc.push(ReprAlign(literal)),
                        Err(message) => {
                            err_span = value.span;
                            literal_error = Some(message)
                        }
                    };
                } else if name == sym::packed {
                    recognised = true;
                    match parse_alignment(&value.kind) {
                        Ok(literal) => acc.push(ReprPacked(literal)),
                        Err(message) => {
                            err_span = value.span;
                            literal_error = Some(message)
                        }
                    };
                } else if matches!(name, sym::Rust | sym::C | sym::simd | sym::transparent)
                    || int_type_of_word(name).is_some()
                {
                    recognised = true;
                    sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoParen {
                        span: item.span(),
                        name: name.to_ident_string(),
                    });
                }
                if let Some(literal_error) = literal_error {
                    sess.dcx().emit_err(session_diagnostics::InvalidReprGeneric {
                        span: err_span,
                        repr_arg: name.to_ident_string(),
                        error_part: literal_error,
                    });
                }
            } else if let Some(meta_item) = item.meta_item() {
                match &meta_item.kind {
                    MetaItemKind::NameValue(value) => {
                        if meta_item.has_name(sym::align) || meta_item.has_name(sym::packed) {
                            let name = meta_item.name_or_empty().to_ident_string();
                            recognised = true;
                            sess.dcx().emit_err(session_diagnostics::IncorrectReprFormatGeneric {
                                span: item.span(),
                                repr_arg: &name,
                                cause: IncorrectReprFormatGenericCause::from_lit_kind(
                                    item.span(),
                                    &value.kind,
                                    &name,
                                ),
                            });
                        } else if matches!(
                            meta_item.name_or_empty(),
                            sym::Rust | sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoValue {
                                span: meta_item.span,
                                name: meta_item.name_or_empty().to_ident_string(),
                            });
                        }
                    }
                    MetaItemKind::List(nested_items) => {
                        if meta_item.has_name(sym::align) {
                            recognised = true;
                            if let [nested_item] = nested_items.as_slice() {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatExpectInteger {
                                        span: nested_item.span(),
                                    },
                                );
                            } else {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatAlignOneArg {
                                        span: meta_item.span,
                                    },
                                );
                            }
                        } else if meta_item.has_name(sym::packed) {
                            recognised = true;
                            if let [nested_item] = nested_items.as_slice() {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatPackedExpectInteger {
                                        span: nested_item.span(),
                                    },
                                );
                            } else {
                                sess.dcx().emit_err(
                                    session_diagnostics::IncorrectReprFormatPackedOneOrZeroArg {
                                        span: meta_item.span,
                                    },
                                );
                            }
                        } else if matches!(
                            meta_item.name_or_empty(),
                            sym::Rust | sym::C | sym::simd | sym::transparent
                        ) || int_type_of_word(meta_item.name_or_empty()).is_some()
                        {
                            recognised = true;
                            sess.dcx().emit_err(session_diagnostics::InvalidReprHintNoParen {
                                span: meta_item.span,
                                name: meta_item.name_or_empty().to_ident_string(),
                            });
                        }
                    }
                    _ => (),
                }
            }
            if !recognised {
                // Not a word we recognize. This will be caught and reported by
                // the `check_mod_attrs` pass, but this pass doesn't always run
                // (e.g. if we only pretty-print the source), so we have to gate
                // the `span_delayed_bug` call as follows:
                if sess.opts.pretty.map_or(true, |pp| pp.needs_analysis()) {
                    dcx.span_delayed_bug(item.span(), "unrecognized representation hint");
                }
            }
        }
    }
    acc
}

fn int_type_of_word(s: Symbol) -> Option<IntType> {
    use rustc_attr_data_structures::IntType::*;

    match s {
        sym::i8 => Some(SignedInt(ast::IntTy::I8)),
        sym::u8 => Some(UnsignedInt(ast::UintTy::U8)),
        sym::i16 => Some(SignedInt(ast::IntTy::I16)),
        sym::u16 => Some(UnsignedInt(ast::UintTy::U16)),
        sym::i32 => Some(SignedInt(ast::IntTy::I32)),
        sym::u32 => Some(UnsignedInt(ast::UintTy::U32)),
        sym::i64 => Some(SignedInt(ast::IntTy::I64)),
        sym::u64 => Some(UnsignedInt(ast::UintTy::U64)),
        sym::i128 => Some(SignedInt(ast::IntTy::I128)),
        sym::u128 => Some(UnsignedInt(ast::UintTy::U128)),
        sym::isize => Some(SignedInt(ast::IntTy::Isize)),
        sym::usize => Some(UnsignedInt(ast::UintTy::Usize)),
        _ => None,
    }
}

pub fn parse_alignment(node: &ast::LitKind) -> Result<Align, &'static str> {
    if let ast::LitKind::Int(literal, ast::LitIntType::Unsuffixed) = node {
        // `Align::from_bytes` accepts 0 as an input, check is_power_of_two() first
        if literal.get().is_power_of_two() {
            // Only possible error is larger than 2^29
            literal
                .get()
                .try_into()
                .ok()
                .and_then(|v| Align::from_bytes(v).ok())
                .ok_or("larger than 2^29")
        } else {
            Err("not a power of two")
        }
    } else {
        Err("not an unsuffixed integer")
    }
}
