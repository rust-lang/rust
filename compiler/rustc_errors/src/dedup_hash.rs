use std::hash::Hash;

use rustc_data_structures::stable_hash::StableHasher;
use rustc_serialize::Encoder;
use rustc_span::def_id::{CrateNum, DefId, DefIndex};
use rustc_span::{ByteSymbol, ExpnId, Span, SpanEncoder, Symbol, SyntaxContext};

/// Hasher used to determine if two diagnostics are the same and should be deduplicated. Used by
/// `DiagCtxtInner::emitted_diagnostics`. It's a stable hash with one special behaviour: parents of
/// spans within the diagnostics are ignored so that incremental and non-incremental compilation
/// get the same behaviour.
///
/// Although this has nothing to do with encoding data to/from file, it is implemented on top of
/// `Encoder`/`SpanEncoder` because they provide traversals of all the relevant types used within
/// diagnostics. There is no corresponding decoder.
pub(crate) struct DedupHashEncoder(pub(crate) StableHasher);

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(
            #[inline]
            fn $name(&mut self, value: $ty) {
                value.hash(&mut self.0)
            }
        )*
    }
}

impl Encoder for DedupHashEncoder {
    encoder_methods! {
        emit_usize(usize);
        emit_u128(u128);
        emit_u64(u64);
        emit_u32(u32);
        emit_u16(u16);
        emit_u8(u8);

        emit_isize(isize);
        emit_i128(i128);
        emit_i64(i64);
        emit_i32(i32);
        emit_i16(i16);

        emit_raw_bytes(&[u8]);
    }
}

impl SpanEncoder for DedupHashEncoder {
    fn encode_span(&mut self, span: Span) {
        // The raison d'être of `DedupHashEncoder` is to ignore `parent` here.
        span.with_parent(None).hash(&mut self.0);
    }

    encoder_methods! {
        encode_symbol(Symbol);
        encode_byte_symbol(ByteSymbol);
        encode_expn_id(ExpnId);
        encode_syntax_context(SyntaxContext);
        encode_crate_num(CrateNum);
        encode_def_index(DefIndex);
        encode_def_id(DefId);
    }
}
