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

use rbml::writer::Encoder as RbmlEncoder;
use rbml::reader::Decoder as RbmlDecoder;
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

    fn encode_ty(&self, rbml_w: &mut RbmlEncoder, t: ty::Ty<'tcx>) {
        encoder::write_type(self, rbml_w, t);
    }

    fn encode_substs(&self, rbml_w: &mut RbmlEncoder, substs: &Substs<'tcx>) {
        let ty_str_ctxt = &tyencode::ctxt {
            diag: self.diag,
            ds: encoder::def_to_string,
            tcx: self.tcx,
            abbrevs: &self.type_abbrevs
        };
        tyencode::enc_substs(rbml_w, ty_str_ctxt, substs);
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

    fn decode_ty(&self, rbml_r: &mut RbmlDecoder) -> ty::Ty<'tcx> {
        let def_id_convert = &mut |did| {
            decoder::translate_def_id(self.crate_metadata, did)
        };

        let starting_position = rbml_r.position();

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
        rbml_r.advance(end_position - starting_position);
        ty
    }

    fn decode_substs(&self, rbml_r: &mut RbmlDecoder) -> Substs<'tcx> {
        let def_id_convert = &mut |did| {
            decoder::translate_def_id(self.crate_metadata, did)
        };

        let starting_position = rbml_r.position();

        let mut ty_decoder = TyDecoder::new(
            self.crate_metadata.data.as_slice(),
            self.crate_metadata.cnum,
            starting_position,
            self.tcx,
            def_id_convert);

        let substs = ty_decoder.parse_substs();

        let end_position = ty_decoder.position();

        rbml_r.advance(end_position - starting_position);
        substs
    }

    fn translate_def_id(&self, def_id: DefId) -> DefId {
        decoder::translate_def_id(self.crate_metadata, def_id)
    }
}
