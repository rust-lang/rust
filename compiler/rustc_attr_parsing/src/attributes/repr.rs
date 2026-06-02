use rustc_abi::{Align, Size};
use rustc_ast::{IntTy, LitIntType, LitKind, UintTy};
use rustc_feature::AttributeStability;
use rustc_hir::attrs::IntType::{SignedInt, UnsignedInt};
use rustc_hir::attrs::ReprAttr;

use super::prelude::*;
use crate::session_diagnostics;

/// Parse #[repr(...)] forms.
///
/// Valid repr contents:
/// * any of the primitive integral type names to specify enum discriminant type
/// * `Rust`, to use the default `Rust` layout of the type
/// * `C`, to use the same layout for the type that C would use
/// * `align(...)`, to change the alignment requirements of the type
/// * `packed`, to remove padding
/// * `transparent`, to delegate representation concerns to the only non-ZST field.
pub(crate) struct ReprParser;

impl CombineAttributeParser for ReprParser {
    type Item = (ReprAttr, Span);
    const PATH: &[Symbol] = &[sym::repr];
    const CONVERT: ConvertFn<Self::Item> =
        |items, first_span| AttributeKind::Repr { reprs: items, first_span };
    const TEMPLATE: AttributeTemplate = template!(
        List: &["C", "Rust", "transparent", "align(...)", "packed(...)", "<integer type>"],
        "https://doc.rust-lang.org/reference/type-layout.html#representations"
    );

    fn extend(
        cx: &mut AcceptContext<'_, '_>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let Some(list) = cx.expect_list(args, cx.attr_span) else {
            return vec![];
        };

        if list.is_empty() {
            let attr_span = cx.attr_span;
            cx.adcx().warn_empty_attribute(attr_span);
            return vec![];
        }

        let mut reprs = Vec::new();
        for param in list.mixed() {
            let Some(item) = param.meta_item() else {
                cx.adcx().expected_identifier(param.span());
                continue;
            };
            reprs.extend(parse_repr(cx, &item).map(|r| (r, param.span())));
        }
        reprs
    }

    //FIXME Still checked fully in `check_attr.rs`
    //This one is slightly more complicated because the allowed targets depend on the arguments
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const STABILITY: AttributeStability = AttributeStability::Stable;
}

fn parse_repr(cx: &mut AcceptContext<'_, '_>, param: &MetaItemParser) -> Option<ReprAttr> {
    use ReprAttr::*;

    macro_rules! no_args {
        ($constructor: expr) => {{
            cx.expect_no_args(param.args())?;
            Some($constructor)
        }};
    }

    match param.path().word_sym() {
        Some(sym::align) => {
            let l = cx.expect_list(param.args(), param.span())?;
            parse_repr_align(cx, l, AlignKind::Align)
        }
        Some(sym::packed) => match param.args() {
            ArgParser::NoArgs => Some(ReprPacked(Align::ONE)),
            ArgParser::List(l) => parse_repr_align(cx, l, AlignKind::Packed),
            ArgParser::NameValue(_) => {
                cx.adcx().expected_list_or_no_args(param.span());
                None
            }
        },
        Some(sym::Rust) => no_args!(ReprRust),
        Some(sym::C) => no_args!(ReprC),
        Some(sym::simd) => no_args!(ReprSimd),
        Some(sym::transparent) => no_args!(ReprTransparent),
        Some(sym::i8) => no_args!(ReprInt(SignedInt(IntTy::I8))),
        Some(sym::u8) => no_args!(ReprInt(UnsignedInt(UintTy::U8))),
        Some(sym::i16) => no_args!(ReprInt(SignedInt(IntTy::I16))),
        Some(sym::u16) => no_args!(ReprInt(UnsignedInt(UintTy::U16))),
        Some(sym::i32) => no_args!(ReprInt(SignedInt(IntTy::I32))),
        Some(sym::u32) => no_args!(ReprInt(UnsignedInt(UintTy::U32))),
        Some(sym::i64) => no_args!(ReprInt(SignedInt(IntTy::I64))),
        Some(sym::u64) => no_args!(ReprInt(UnsignedInt(UintTy::U64))),
        Some(sym::i128) => no_args!(ReprInt(SignedInt(IntTy::I128))),
        Some(sym::u128) => no_args!(ReprInt(UnsignedInt(UintTy::U128))),
        Some(sym::isize) => no_args!(ReprInt(SignedInt(IntTy::Isize))),
        Some(sym::usize) => no_args!(ReprInt(UnsignedInt(UintTy::Usize))),
        _ => {
            cx.adcx().expected_specific_argument(
                param.span(),
                &[
                    sym::align,
                    sym::packed,
                    sym::Rust,
                    sym::C,
                    sym::simd,
                    sym::transparent,
                    sym::i8,
                    sym::u8,
                    sym::i16,
                    sym::u16,
                    sym::i32,
                    sym::u32,
                    sym::i64,
                    sym::u64,
                    sym::i128,
                    sym::u128,
                    sym::isize,
                    sym::usize,
                ],
            );
            None
        }
    }
}

enum AlignKind {
    Packed,
    Align,
}

fn parse_repr_align(
    cx: &mut AcceptContext<'_, '_>,
    list: &MetaItemListParser,
    align_kind: AlignKind,
) -> Option<ReprAttr> {
    let Some(align) = list.as_single() else {
        cx.adcx().expected_single_argument(list.span, list.len());
        return None;
    };

    let Some(lit) = align.as_lit() else {
        cx.adcx().expected_integer_literal(align.span());
        return None;
    };

    match parse_alignment(&lit.kind, cx) {
        Ok(literal) => Some(match align_kind {
            AlignKind::Packed => ReprAttr::ReprPacked(literal),
            AlignKind::Align => ReprAttr::ReprAlign(literal),
        }),
        Err(message) => {
            cx.emit_err(session_diagnostics::InvalidAlignmentValue {
                span: lit.span,
                error_part: message,
            });
            None
        }
    }
}

fn parse_alignment(node: &LitKind, cx: &AcceptContext<'_, '_>) -> Result<Align, String> {
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

/// Parse #[align(N)].
#[derive(Default)]
pub(crate) struct RustcAlignParser(Option<(Align, Span)>);

impl RustcAlignParser {
    const PATH: &[Symbol] = &[sym::rustc_align];
    const TEMPLATE: AttributeTemplate = template!(List: &["<alignment in bytes>"]);

    fn parse(&mut self, cx: &mut AcceptContext<'_, '_>, args: &ArgParser) {
        let Some(list) = cx.expect_list(args, cx.attr_span) else {
            return;
        };

        let Some(align) = cx.expect_single(list) else {
            return;
        };

        let Some(lit) = align.as_lit() else {
            cx.adcx().expected_integer_literal(align.span());
            return;
        };

        match parse_alignment(&lit.kind, cx) {
            Ok(literal) => self.0 = Ord::max(self.0, Some((literal, cx.attr_span))),
            Err(message) => {
                cx.emit_err(session_diagnostics::InvalidAlignmentValue {
                    span: lit.span,
                    error_part: message,
                });
            }
        }
    }
}

impl AttributeParser for RustcAlignParser {
    const ATTRIBUTES: AcceptMapping<Self> =
        &[(Self::PATH, Self::TEMPLATE, unstable!(fn_align), Self::parse)];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })), // `#[align]` is inherited from trait methods
        Allow(Target::ForeignFn),
    ]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        let (align, span) = self.0?;
        Some(AttributeKind::RustcAlign { align, span })
    }
}

#[derive(Default)]
pub(crate) struct RustcAlignStaticParser(RustcAlignParser);

impl RustcAlignStaticParser {
    const PATH: &[Symbol] = &[sym::rustc_align_static];
    const TEMPLATE: AttributeTemplate = RustcAlignParser::TEMPLATE;

    fn parse(&mut self, cx: &mut AcceptContext<'_, '_>, args: &ArgParser) {
        self.0.parse(cx, args)
    }
}

impl AttributeParser for RustcAlignStaticParser {
    const ATTRIBUTES: AcceptMapping<Self> =
        &[(Self::PATH, Self::TEMPLATE, unstable!(static_align), Self::parse)];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Static), Allow(Target::ForeignStatic)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_>) -> Option<AttributeKind> {
        let (align, span) = self.0.0?;
        Some(AttributeKind::RustcAlign { align, span })
    }
}
