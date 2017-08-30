// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use fmt_macros::{Parser, Piece, Position};

use hir::def_id::DefId;
use ty::{self, TyCtxt};
use util::common::ErrorReported;
use util::nodemap::FxHashMap;

use syntax_pos::Span;
use syntax_pos::symbol::InternedString;

pub struct OnUnimplementedFormatString(InternedString);
pub struct OnUnimplementedInfo {
    pub label: OnUnimplementedFormatString
}

impl<'a, 'gcx, 'tcx> OnUnimplementedInfo {
    pub fn of_item(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                   trait_def_id: DefId,
                   impl_def_id: DefId,
                   span: Span)
                   -> Result<Option<Self>, ErrorReported>
    {
        let attrs = tcx.get_attrs(impl_def_id);

        let attr = if let Some(item) =
            attrs.into_iter().find(|a| a.check_name("rustc_on_unimplemented"))
        {
            item
        } else {
            return Ok(None);
        };

        let span = attr.span.substitute_dummy(span);
        if let Some(label) = attr.value_str() {
            Ok(Some(OnUnimplementedInfo {
                label: OnUnimplementedFormatString::try_parse(
                    tcx, trait_def_id, label.as_str(), span)?
            }))
        } else {
            struct_span_err!(
                tcx.sess, span, E0232,
                "this attribute must have a value")
                .span_label(attr.span, "attribute requires a value")
                .note(&format!("eg `#[rustc_on_unimplemented = \"foo\"]`"))
                .emit();
            Err(ErrorReported)
        }
    }
}

impl<'a, 'gcx, 'tcx> OnUnimplementedFormatString {
    pub fn try_parse(tcx: TyCtxt<'a, 'gcx, 'tcx>,
                     trait_def_id: DefId,
                     from: InternedString,
                     err_sp: Span)
                     -> Result<Self, ErrorReported>
    {
        let result = OnUnimplementedFormatString(from);
        result.verify(tcx, trait_def_id, err_sp)?;
        Ok(result)
    }

    fn verify(&self,
              tcx: TyCtxt<'a, 'gcx, 'tcx>,
              trait_def_id: DefId,
              span: Span)
              -> Result<(), ErrorReported>
    {
        let name = tcx.item_name(trait_def_id).as_str();
        let generics = tcx.generics_of(trait_def_id);
        let parser = Parser::new(&self.0);
        let types = &generics.types;
        let mut result = Ok(());
        for token in parser {
            match token {
                Piece::String(_) => (), // Normal string, no need to check it
                Piece::NextArgument(a) => match a.position {
                    // `{Self}` is allowed
                    Position::ArgumentNamed(s) if s == "Self" => (),
                    // `{ThisTraitsName}` is allowed
                    Position::ArgumentNamed(s) if s == name => (),
                    // So is `{A}` if A is a type parameter
                    Position::ArgumentNamed(s) => match types.iter().find(|t| {
                        t.name == s
                    }) {
                        Some(_) => (),
                        None => {
                            span_err!(tcx.sess, span, E0230,
                                      "there is no type parameter \
                                       {} on trait {}",
                                      s, name);
                            result = Err(ErrorReported);
                        }
                    },
                    // `{:1}` and `{}` are not to be used
                    Position::ArgumentIs(_) => {
                        span_err!(tcx.sess, span, E0231,
                                  "only named substitution \
                                   parameters are allowed");
                        result = Err(ErrorReported);
                    }
                }
            }
        }

        result
    }

    pub fn format(&self,
                  tcx: TyCtxt<'a, 'gcx, 'tcx>,
                  trait_ref: ty::TraitRef<'tcx>)
                  -> String
    {
        let name = tcx.item_name(trait_ref.def_id).as_str();
        let trait_str = tcx.item_path_str(trait_ref.def_id);
        let generics = tcx.generics_of(trait_ref.def_id);
        let generic_map = generics.types.iter().map(|param| {
            (param.name.as_str().to_string(),
             trait_ref.substs.type_for_def(param).to_string())
        }).collect::<FxHashMap<String, String>>();

        let parser = Parser::new(&self.0);
        parser.map(|p| {
            match p {
                Piece::String(s) => s,
                Piece::NextArgument(a) => match a.position {
                    Position::ArgumentNamed(s) => match generic_map.get(s) {
                        Some(val) => val,
                        None if s == name => {
                            &trait_str
                        }
                        None => {
                            bug!("broken on_unimplemented {:?} for {:?}: \
                                  no argument matching {:?}",
                                 self.0, trait_ref, s)
                        }
                    },
                    _ => {
                        bug!("broken on_unimplemented {:?} - bad \
                              format arg", self.0)
                    }
                }
            }
        }).collect()
    }
}
