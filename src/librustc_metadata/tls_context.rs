// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This module provides implementations for the thread-local encoding and
// decoding context traits in rustc::middle::cstore::tls.

use rbml::opaque::Encoder as OpaqueEncoder;
use rbml::opaque::Decoder as OpaqueDecoder;
use rustc::middle::cstore::tls;
use rustc::middle::def_id::DefId;
use rustc::middle::subst::Substs;
use rustc::middle::ty;

use decoder::{self, Cmd};
use encoder;
use tydecode::TyDecoder;
use tyencode;

impl<'a, 'tcx: 'a> tls::EncodingContext<'tcx> for encoder::EncodeContext<'a, 'tcx> {

    fn tcx<'s>(&'s self) -> &'s ty::ctxt<'tcx> {
        &self.tcx
    }

    fn encode_ty(&self, encoder: &mut OpaqueEncoder, t: ty::Ty<'tcx>) {
        tyencode::enc_ty(encoder.cursor, &self.ty_str_ctxt(), t);
    }

    fn encode_substs(&self, encoder: &mut OpaqueEncoder, substs: &Substs<'tcx>) {
        tyencode::enc_substs(encoder.cursor, &self.ty_str_ctxt(), substs);
    }
}

pub struct DecodingContext<'a, 'tcx: 'a> {
    pub crate_metadata: Cmd<'a>,
    pub tcx: &'a ty::ctxt<'tcx>,
}

impl<'a, 'tcx: 'a> tls::DecodingContext<'tcx> for DecodingContext<'a, 'tcx> {

    fn tcx<'s>(&'s self) -> &'s ty::ctxt<'tcx> {
        &self.tcx
    }

    fn decode_ty(&self, decoder: &mut OpaqueDecoder) -> ty::Ty<'tcx> {
        let def_id_convert = &mut |did| {
            decoder::translate_def_id(self.crate_metadata, did)
        };

        let starting_position = decoder.position();

        let mut ty_decoder = TyDecoder::new(
            self.crate_metadata.data.as_slice(),
            self.crate_metadata.cnum,
            starting_position,
            self.tcx,
            def_id_convert);

        let ty = ty_decoder.parse_ty();

        let end_position = ty_decoder.position();

        // We can just reuse the tydecode implementation for parsing types, but
        // we have to make sure to leave the rbml reader at the position just
        // after the type.
        decoder.advance(end_position - starting_position);
        ty
    }

    fn decode_substs(&self, decoder: &mut OpaqueDecoder) -> Substs<'tcx> {
        let def_id_convert = &mut |did| {
            decoder::translate_def_id(self.crate_metadata, did)
        };

        let starting_position = decoder.position();

        let mut ty_decoder = TyDecoder::new(
            self.crate_metadata.data.as_slice(),
            self.crate_metadata.cnum,
            starting_position,
            self.tcx,
            def_id_convert);

        let substs = ty_decoder.parse_substs();

        let end_position = ty_decoder.position();

        decoder.advance(end_position - starting_position);
        substs
    }

    fn translate_def_id(&self, def_id: DefId) -> DefId {
        decoder::translate_def_id(self.crate_metadata, def_id)
    }
}
