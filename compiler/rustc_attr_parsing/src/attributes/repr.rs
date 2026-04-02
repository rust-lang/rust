use rustc_abi::{Align, Size};
use rustc_ast::{IntTy, LitIntType, LitKind, UintTy};
use rustc_hir::attrs::{AttrConstResolved, AttrIntValue, IntType, ReprAttr};
use rustc_hir::def::{DefKind, Res};
use rustc_session::parse::feature_err;

use super::prelude::*;
use crate::ShouldEmit;
use crate::session_diagnostics::{
    self, AttrConstGenericNotSupported, AttrConstPathNotConst, IncorrectReprFormatGenericCause,
};

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
    const CONVERT: ConvertFn<Self::Item> =
        |items, first_span| AttributeKind::Repr { reprs: items, first_span };
    // FIXME(jdonszelmann): never used
    const TEMPLATE: AttributeTemplate = template!(
        List: &["C", "Rust", "transparent", "align(...)", "packed(...)", "<integer type>"],
        "https://doc.rust-lang.org/reference/type-layout.html#representations"
    );

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let mut reprs = Vec::new();

        let Some(list) = args.list() else {
            let attr_span = cx.attr_span;
            cx.adcx().expected_list(attr_span, args);
            return reprs;
        };

        if list.is_empty() {
            let attr_span = cx.attr_span;
            cx.adcx().warn_empty_attribute(attr_span);
            return reprs;
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

    //FIXME Still checked fully in `check_attr.rs`
    //This one is slightly more complicated because the allowed targets depend on the arguments
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
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

fn parse_repr<S: Stage>(cx: &AcceptContext<'_, '_, S>, param: &MetaItemParser) -> Option<ReprAttr> {
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

        (Some(sym::packed), ArgParser::NoArgs) => Some(ReprPacked(AttrIntValue::Lit(1))),
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

enum AlignmentParseError {
    Message(String),
    AlreadyErrored,
}

fn parse_repr_align<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    list: &MetaItemListParser,
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
                cx.emit_err(session_diagnostics::IncorrectReprFormatAlignOneArg {
                    span: param_span,
                });
            }
        }

        return None;
    };

    match parse_alignment_or_const_path(
        cx,
        align,
        match align_kind {
            Packed => "repr(packed)",
            Align => "repr(align)",
        },
    ) {
        Ok(value) => Some(match align_kind {
            AlignKind::Packed => ReprAttr::ReprPacked(value),
            AlignKind::Align => ReprAttr::ReprAlign(value),
        }),
        Err(AlignmentParseError::Message(message)) => {
            cx.emit_err(session_diagnostics::InvalidReprGeneric {
                span: align.span(),
                repr_arg: match align_kind {
                    Packed => "packed".to_string(),
                    Align => "align".to_string(),
                },
                error_part: message,
            });
            None
        }
        Err(AlignmentParseError::AlreadyErrored) => None,
    }
}

fn parse_alignment<S: Stage>(
    node: &LitKind,
    cx: &AcceptContext<'_, '_, S>,
) -> Result<Align, String> {
    let LitKind::Int(literal, LitIntType::Unsuffixed) = node else {
        return Err("not an unsuffixed integer".to_string());
    };

    // `Align::from_bytes` accepts 0 as a valid input,
    // so we check if its a power of two first
    if !literal.get().is_power_of_two() {
        return Err("not a power of two".to_string());
    }
    // lit must be < 2^29
    let align = literal
        .get()
        .try_into()
        .ok()
        .and_then(|a| Align::from_bytes(a).ok())
        .ok_or("larger than 2^29".to_string())?;

    // alignment must not be larger than the pointer width (`isize::MAX`)
    let max = Size::from_bits(cx.sess.target.pointer_width).signed_int_max() as u64;
    if align.bytes() > max {
        return Err(format!(
            "alignment larger than `isize::MAX` bytes ({max} for the current target)"
        ));
    }
    Ok(align)
}

fn parse_alignment_or_const_path<S: Stage>(
    cx: &AcceptContext<'_, '_, S>,
    arg: &MetaItemOrLitParser,
    attr_name: &'static str,
) -> Result<AttrIntValue, AlignmentParseError> {
    if let Some(lit) = arg.lit() {
        return parse_alignment(&lit.kind, cx)
            .map(|align| AttrIntValue::Lit(u128::from(align.bytes())))
            .map_err(AlignmentParseError::Message);
    }

    let Some(meta) = arg.meta_item() else {
        return Err(AlignmentParseError::Message("not an unsuffixed integer".to_string()));
    };

    if !matches!(meta.args(), ArgParser::NoArgs) {
        return Err(AlignmentParseError::Message("not an unsuffixed integer".to_string()));
    }

    if let Some(features) = cx.features_option()
        && !features.const_attr_paths()
        && !meta.span().allows_unstable(sym::const_attr_paths)
    {
        feature_err(
            cx.sess(),
            sym::const_attr_paths,
            meta.span(),
            "const item paths in builtin attributes are experimental",
        )
        .emit();
        return Err(AlignmentParseError::AlreadyErrored);
    }

    let Some(resolution) = cx.attr_const_resolution(meta.path().span()) else {
        // `parse_limited(sym::repr)` runs before lowering for callers that only care whether
        // `repr(packed(...))` exists at all.
        if matches!(cx.stage.should_emit(), ShouldEmit::Nothing) {
            return Ok(AttrIntValue::Lit(1));
        }
        return Err(AlignmentParseError::Message("not an unsuffixed integer".to_string()));
    };

    match resolution {
        AttrConstResolved::Resolved(Res::Def(DefKind::Const { .. }, def_id)) => {
            Ok(AttrIntValue::Const { def_id, span: meta.path().span() })
        }
        AttrConstResolved::Resolved(Res::Def(DefKind::ConstParam, _)) => {
            cx.emit_err(AttrConstGenericNotSupported { span: meta.path().span(), attr_name });
            Err(AlignmentParseError::AlreadyErrored)
        }
        AttrConstResolved::Resolved(res) => {
            cx.emit_err(AttrConstPathNotConst {
                span: meta.path().span(),
                attr_name,
                thing: res.descr(),
            });
            Err(AlignmentParseError::AlreadyErrored)
        }
        AttrConstResolved::Error => Err(AlignmentParseError::AlreadyErrored),
    }
}

/// Parse #[align(N)].
#[derive(Default)]
pub(crate) struct RustcAlignParser(ThinVec<(AttrIntValue, Span)>);

impl RustcAlignParser {
    const PATH: &[Symbol] = &[sym::rustc_align];
    const TEMPLATE: AttributeTemplate = template!(List: &["<alignment in bytes>"]);

    fn parse<S: Stage>(&mut self, cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) {
        match args {
            ArgParser::NoArgs | ArgParser::NameValue(_) => {
                let attr_span = cx.attr_span;
                cx.adcx().expected_list(attr_span, args);
            }
            ArgParser::List(list) => {
                let Some(align) = list.single() else {
                    cx.adcx().expected_single_argument(list.span, list.len());
                    return;
                };

                match parse_alignment_or_const_path(cx, align, "rustc_align") {
                    Ok(literal) => self.0.push((literal, cx.attr_span)),
                    Err(AlignmentParseError::Message(message)) => {
                        cx.emit_err(session_diagnostics::InvalidAlignmentValue {
                            span: align.span(),
                            error_part: message,
                        });
                    }
                    Err(AlignmentParseError::AlreadyErrored) => {}
                }
            }
        }
    }
}

impl<S: Stage> AttributeParser<S> for RustcAlignParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(Self::PATH, Self::TEMPLATE, Self::parse)];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })), // `#[align]` is inherited from trait methods
        Allow(Target::ForeignFn),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        (!self.0.is_empty()).then_some(AttributeKind::RustcAlign { aligns: self.0 })
    }
}

#[derive(Default)]
pub(crate) struct RustcAlignStaticParser(RustcAlignParser);

impl RustcAlignStaticParser {
    const PATH: &[Symbol] = &[sym::rustc_align_static];
    const TEMPLATE: AttributeTemplate = RustcAlignParser::TEMPLATE;

    fn parse<S: Stage>(&mut self, cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) {
        self.0.parse(cx, args)
    }
}

impl<S: Stage> AttributeParser<S> for RustcAlignStaticParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(Self::PATH, Self::TEMPLATE, Self::parse)];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Static), Allow(Target::ForeignStatic)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        (!self.0.0.is_empty()).then_some(AttributeKind::RustcAlign { aligns: self.0.0 })
    }
}
