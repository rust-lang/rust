use rustc_abi::Align;
use rustc_ast::{IntTy, LitIntType, LitKind, UintTy};
use rustc_attr_data_structures::{AttributeKind, IntType, ReprAttr};
use rustc_span::{Span, Symbol, sym};

use super::{CombineAttributeParser, ConvertFn};
use crate::context::AcceptContext;
use crate::parser::{ArgParser, MetaItemListParser, MetaItemParser};
use crate::session_diagnostics;
use crate::session_diagnostics::IncorrectReprFormatGenericCause;

/// Parse #[repr(...)] forms.
///
/// Valid repr contents: any of the primitive integral type names (see
/// `int_type_of_word`, below) to specify enum discriminant type; `C`, to use
/// the same discriminant size that the corresponding C enum would or C
/// structure layout, `packed` to remove padding, and `transparent` to delegate representation
/// concerns to the only non-ZST field.
// FIXME(jdonszelmann): is a vec the right representation here even? isn't it just a struct?
pub(crate) struct ReprParser;

impl CombineAttributeParser for ReprParser {
    type Item = (ReprAttr, Span);
    const PATH: &'static [rustc_span::Symbol] = &[sym::repr];
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Repr;

    fn extend<'a>(
        cx: &'a AcceptContext<'a>,
        args: &'a ArgParser<'a>,
    ) -> impl IntoIterator<Item = Self::Item> + 'a {
        let mut reprs = Vec::new();

        let Some(list) = args.list() else {
            return reprs;
        };

        if list.is_empty() {
            // this is so validation can emit a lint
            reprs.push((ReprAttr::ReprEmpty, cx.attr_span));
        }

        for param in list.mixed() {
            if let Some(_) = param.lit() {
                cx.emit_err(session_diagnostics::ReprIdent { span: cx.attr_span });
                continue;
            }

            reprs.extend(
                param.meta_item().and_then(|mi| parse_repr(cx, &mi)).map(|r| (r, param.span())),
            );
        }

        reprs
    }
}

macro_rules! int_pat {
    () => {
        sym::i8
            | sym::u8
            | sym::i16
            | sym::u16
            | sym::i32
            | sym::u32
            | sym::i64
            | sym::u64
            | sym::i128
            | sym::u128
            | sym::isize
            | sym::usize
    };
}

fn int_type_of_word(s: Symbol) -> Option<IntType> {
    use IntType::*;

    match s {
        sym::i8 => Some(SignedInt(IntTy::I8)),
        sym::u8 => Some(UnsignedInt(UintTy::U8)),
        sym::i16 => Some(SignedInt(IntTy::I16)),
        sym::u16 => Some(UnsignedInt(UintTy::U16)),
        sym::i32 => Some(SignedInt(IntTy::I32)),
        sym::u32 => Some(UnsignedInt(UintTy::U32)),
        sym::i64 => Some(SignedInt(IntTy::I64)),
        sym::u64 => Some(UnsignedInt(UintTy::U64)),
        sym::i128 => Some(SignedInt(IntTy::I128)),
        sym::u128 => Some(UnsignedInt(UintTy::U128)),
        sym::isize => Some(SignedInt(IntTy::Isize)),
        sym::usize => Some(UnsignedInt(UintTy::Usize)),
        _ => None,
    }
}

fn parse_repr(cx: &AcceptContext<'_>, param: &MetaItemParser<'_>) -> Option<ReprAttr> {
    use ReprAttr::*;

    // FIXME(jdonszelmann): invert the parsing here to match on the word first and then the
    // structure.
    let (ident, args) = param.word_or_empty();

    match (ident.name, args) {
        (sym::align, ArgParser::NoArgs) => {
            cx.emit_err(session_diagnostics::InvalidReprAlignNeedArg { span: ident.span });
            None
        }
        (sym::align, ArgParser::List(l)) => parse_repr_align(cx, l, param.span(), AlignKind::Align),

        (sym::packed, ArgParser::NoArgs) => Some(ReprPacked(Align::ONE)),
        (sym::packed, ArgParser::List(l)) => {
            parse_repr_align(cx, l, param.span(), AlignKind::Packed)
        }

        (sym::align | sym::packed, ArgParser::NameValue(l)) => {
            cx.emit_err(session_diagnostics::IncorrectReprFormatGeneric {
                span: param.span(),
                // FIXME(jdonszelmann) can just be a string in the diag type
                repr_arg: &ident.to_string(),
                cause: IncorrectReprFormatGenericCause::from_lit_kind(
                    param.span(),
                    &l.value_as_lit().kind,
                    ident.name.as_str(),
                ),
            });
            None
        }

        (sym::Rust, ArgParser::NoArgs) => Some(ReprRust),
        (sym::C, ArgParser::NoArgs) => Some(ReprC),
        (sym::simd, ArgParser::NoArgs) => Some(ReprSimd),
        (sym::transparent, ArgParser::NoArgs) => Some(ReprTransparent),
        (i @ int_pat!(), ArgParser::NoArgs) => {
            // int_pat!() should make sure it always parses
            Some(ReprInt(int_type_of_word(i).unwrap()))
        }

        (
            sym::Rust | sym::C | sym::simd | sym::transparent | int_pat!(),
            ArgParser::NameValue(_),
        ) => {
            cx.emit_err(session_diagnostics::InvalidReprHintNoValue {
                span: param.span(),
                name: ident.to_string(),
            });
            None
        }
        (sym::Rust | sym::C | sym::simd | sym::transparent | int_pat!(), ArgParser::List(_)) => {
            cx.emit_err(session_diagnostics::InvalidReprHintNoParen {
                span: param.span(),
                name: ident.to_string(),
            });
            None
        }

        _ => {
            cx.emit_err(session_diagnostics::UnrecognizedReprHint { span: param.span() });
            None
        }
    }
}

enum AlignKind {
    Packed,
    Align,
}

fn parse_repr_align(
    cx: &AcceptContext<'_>,
    list: &MetaItemListParser<'_>,
    param_span: Span,
    align_kind: AlignKind,
) -> Option<ReprAttr> {
    use AlignKind::*;

    let Some(align) = list.single() else {
        match align_kind {
            Packed => {
                cx.emit_err(session_diagnostics::IncorrectReprFormatPackedOneOrZeroArg {
                    span: param_span,
                });
            }
            Align => {
                cx.dcx().emit_err(session_diagnostics::IncorrectReprFormatAlignOneArg {
                    span: param_span,
                });
            }
        }

        return None;
    };

    let Some(lit) = align.lit() else {
        match align_kind {
            Packed => {
                cx.emit_err(session_diagnostics::IncorrectReprFormatPackedExpectInteger {
                    span: align.span(),
                });
            }
            Align => {
                cx.emit_err(session_diagnostics::IncorrectReprFormatExpectInteger {
                    span: align.span(),
                });
            }
        }

        return None;
    };

    match parse_alignment(&lit.kind) {
        Ok(literal) => Some(match align_kind {
            AlignKind::Packed => ReprAttr::ReprPacked(literal),
            AlignKind::Align => ReprAttr::ReprAlign(literal),
        }),
        Err(message) => {
            cx.emit_err(session_diagnostics::InvalidReprGeneric {
                span: lit.span,
                repr_arg: match align_kind {
                    Packed => "packed".to_string(),
                    Align => "align".to_string(),
                },
                error_part: message,
            });
            None
        }
    }
}

fn parse_alignment(node: &LitKind) -> Result<Align, &'static str> {
    if let LitKind::Int(literal, LitIntType::Unsuffixed) = node {
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
