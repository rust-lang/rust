use rustc_abi::Align;
use rustc_ast::{IntTy, LitIntType, LitKind, UintTy};
use rustc_attr_data_structures::{AttributeKind, IntType, ReprAttr};
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{DUMMY_SP, Span, Symbol, sym};

use super::{CombineAttributeParser, ConvertFn};
use crate::context::{AcceptContext, Stage};
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

impl<S: Stage> CombineAttributeParser<S> for ReprParser {
    type Item = (ReprAttr, Span);
    const PATH: &[Symbol] = &[sym::repr];
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Repr;
    // FIXME(jdonszelmann): never used
    const TEMPLATE: AttributeTemplate = template!(List: "C");

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let mut reprs = Vec::new();

        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
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

fn parse_repr<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    param: &MetaItemParser<'_>,
) -> Option<ReprAttr> {
    use ReprAttr::*;

    // FIXME(jdonszelmann): invert the parsing here to match on the word first and then the
    // structure.
    let (name, ident_span) = if let Some(ident) = param.path().word() {
        (Some(ident.name), ident.span)
    } else {
        (None, DUMMY_SP)
    };

    let args = param.args();

    match (name, args) {
        (Some(sym::align), ArgParser::NoArgs) => {
            cx.emit_err(session_diagnostics::InvalidReprAlignNeedArg { span: ident_span });
            None
        }
        (Some(sym::align), ArgParser::List(l)) => {
            parse_repr_align(cx, l, param.span(), AlignKind::Align)
        }

        (Some(sym::packed), ArgParser::NoArgs) => Some(ReprPacked(Align::ONE)),
        (Some(sym::packed), ArgParser::List(l)) => {
            parse_repr_align(cx, l, param.span(), AlignKind::Packed)
        }

        (Some(name @ sym::align | name @ sym::packed), ArgParser::NameValue(l)) => {
            cx.emit_err(session_diagnostics::IncorrectReprFormatGeneric {
                span: param.span(),
                // FIXME(jdonszelmann) can just be a string in the diag type
                repr_arg: name,
                cause: IncorrectReprFormatGenericCause::from_lit_kind(
                    param.span(),
                    &l.value_as_lit().kind,
                    name,
                ),
            });
            None
        }

        (Some(sym::Rust), ArgParser::NoArgs) => Some(ReprRust),
        (Some(sym::C), ArgParser::NoArgs) => Some(ReprC),
        (Some(sym::simd), ArgParser::NoArgs) => Some(ReprSimd),
        (Some(sym::transparent), ArgParser::NoArgs) => Some(ReprTransparent),
        (Some(name @ int_pat!()), ArgParser::NoArgs) => {
            // int_pat!() should make sure it always parses
            Some(ReprInt(int_type_of_word(name).unwrap()))
        }

        (
            Some(
                name @ sym::Rust
                | name @ sym::C
                | name @ sym::simd
                | name @ sym::transparent
                | name @ int_pat!(),
            ),
            ArgParser::NameValue(_),
        ) => {
            cx.emit_err(session_diagnostics::InvalidReprHintNoValue { span: param.span(), name });
            None
        }
        (
            Some(
                name @ sym::Rust
                | name @ sym::C
                | name @ sym::simd
                | name @ sym::transparent
                | name @ int_pat!(),
            ),
            ArgParser::List(_),
        ) => {
            cx.emit_err(session_diagnostics::InvalidReprHintNoParen { span: param.span(), name });
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

fn parse_repr_align<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
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
