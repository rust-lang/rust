use rustc_abi::{CCharTy, Float, Integer};
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_expand::base::*;
use rustc_parse::exp;
use rustc_span::{Span, Symbol, sym};

use crate::diagnostics;

pub(crate) fn expand_c_type<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    span: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let mut p = cx.new_parser_from_tts(tts);

    ExpandResult::Ready('blk: {
        let ident = match p.parse_ident() {
            Ok(ident) => ident,
            Err(err) => break 'blk DummyResult::any(span, err.emit_err()),
        };

        let _ = p.eat(exp!(Comma));

        if p.token != token::Eof {
            let diag = diagnostics::OnlyOneArgument { span: p.token.span, name: "c_type" };
            break 'blk DummyResult::any(span, cx.dcx().emit_err(diag));
        }

        let signed = |integer| match integer {
            Integer::I8 => sym::i8,
            Integer::I16 => sym::i16,
            Integer::I32 => sym::i32,
            Integer::I64 => sym::i64,
            Integer::I128 => sym::i128,
        };

        let unsigned = |integer| match integer {
            Integer::I8 => sym::u8,
            Integer::I16 => sym::u16,
            Integer::I32 => sym::u32,
            Integer::I64 => sym::u64,
            Integer::I128 => sym::u128,
        };

        fn float(f: Float) -> Symbol {
            match f {
                Float::F16 => sym::f16,
                Float::F32 => sym::f32,
                Float::F64 => sym::f64,
                Float::F128 => sym::f128,
                Float::F16B | Float::X87F80 | Float::PpcF128 => unreachable!("for now"),
            }
        }

        let c_types = &cx.sess.target.c_types;

        let symbol = match ident.name {
            sym::c_char => match c_types.c_char {
                CCharTy::I8 => sym::i8,
                CCharTy::U8 => sym::u8,
            },
            sym::c_schar => sym::i8, // signed(c_types.c_char),
            sym::c_uchar => sym::u8, // unsigned(c_types.c_char),

            sym::c_short => signed(c_types.c_short),
            sym::c_int => signed(c_types.c_int),
            sym::c_long => signed(c_types.c_long),
            sym::c_longlong => signed(c_types.c_longlong),

            sym::c_ushort => unsigned(c_types.c_short),
            sym::c_uint => unsigned(c_types.c_int),
            sym::c_ulong => unsigned(c_types.c_long),
            sym::c_ulonglong => unsigned(c_types.c_longlong),

            sym::c_float => float(c_types.c_float),
            sym::c_double => float(c_types.c_double),

            _ => {
                let diag = diagnostics::BadCType { span: ident.span, name: ident.name };
                break 'blk DummyResult::any(span, cx.dcx().emit_err(diag));
            }
        };

        MacEager::ty(cx.ty_path(cx.std_path(span, &[sym::primitive, symbol])))
    })
}
