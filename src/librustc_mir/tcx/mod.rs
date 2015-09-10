// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use hair::*;
use repr::*;
use std::fmt::{Debug, Formatter, Error};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use self::rustc::middle::const_eval::ConstVal;
use self::rustc::middle::def_id::DefId;
use self::rustc::middle::infer::InferCtxt;
use self::rustc::middle::region::CodeExtent;
use self::rustc::middle::subst::{self, Subst, Substs};
use self::rustc::middle::ty::{self, Ty};
use self::rustc_front::hir;
use self::syntax::ast;
use self::syntax::codemap::Span;
use self::syntax::parse::token::{self, special_idents, InternedString};

extern crate rustc;
extern crate rustc_front;
extern crate syntax;

#[derive(Copy, Clone)]
pub struct Cx<'a,'tcx:'a> {
    pub tcx: &'a ty::ctxt<'tcx>,
    pub infcx: &'a InferCtxt<'a,'tcx>,
}

impl<'a,'tcx> Cx<'a,'tcx> {
    pub fn new(infcx: &'a InferCtxt<'a,'tcx>) -> Cx<'a,'tcx> {
        Cx { tcx: infcx.tcx, infcx: infcx }
    }
}

pub use self::pattern::PatNode;

impl<'a,'tcx:'a> Hair for Cx<'a, 'tcx> {
    type VarId = ast::NodeId;
    type DefId = DefId;
    type AdtDef = ty::AdtDef<'tcx>;
    type Name = ast::Name;
    type InternedString = InternedString;
    type Bytes = Rc<Vec<u8>>;
    type Span = Span;
    type Projection = ty::ProjectionTy<'tcx>;
    type Substs = &'tcx subst::Substs<'tcx>;
    type ClosureSubsts = &'tcx ty::ClosureSubsts<'tcx>;
    type Ty = Ty<'tcx>;
    type Region = ty::Region;
    type CodeExtent = CodeExtent;
    type ConstVal = ConstVal;
    type Pattern = PatNode<'tcx>;
    type Expr = &'tcx hir::Expr;
    type Stmt = &'tcx hir::Stmt;
    type Block = &'tcx hir::Block;
    type InlineAsm = &'tcx hir::InlineAsm;

    fn unit_ty(&mut self) -> Ty<'tcx> {
        self.tcx.mk_nil()
    }

    fn usize_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.usize
    }

    fn usize_literal(&mut self, value: usize) -> Literal<Self> {
        Literal::Value { value: ConstVal::Uint(value as u64) }
    }

    fn bool_ty(&mut self) -> Ty<'tcx> {
        self.tcx.types.bool
    }

    fn true_literal(&mut self) -> Literal<Self> {
        Literal::Value { value: ConstVal::Bool(true) }
    }

    fn false_literal(&mut self) -> Literal<Self> {
        Literal::Value { value: ConstVal::Bool(false) }
    }

    fn partial_eq(&mut self, ty: Ty<'tcx>) -> ItemRef<Self> {
        let eq_def_id = self.tcx.lang_items.eq_trait().unwrap();
        self.cmp_method_ref(eq_def_id, "eq", ty)
    }

    fn partial_le(&mut self, ty: Ty<'tcx>) -> ItemRef<Self> {
        let ord_def_id = self.tcx.lang_items.ord_trait().unwrap();
        self.cmp_method_ref(ord_def_id, "le", ty)
    }

    fn num_variants(&mut self, adt_def: ty::AdtDef<'tcx>) -> usize {
        adt_def.variants.len()
    }

    fn fields(&mut self, adt_def: ty::AdtDef<'tcx>, variant_index: usize) -> Vec<Field<Self>> {
        adt_def.variants[variant_index]
            .fields
            .iter()
            .enumerate()
            .map(|(index, field)| {
                if field.name == special_idents::unnamed_field.name {
                    Field::Indexed(index)
                } else {
                    Field::Named(field.name)
                }
            })
            .collect()
    }

    fn needs_drop(&mut self, ty: Ty<'tcx>, span: Self::Span) -> bool {
        if self.infcx.type_moves_by_default(ty, span) {
            // FIXME(#21859) we should do an add'l check here to determine if
            // any dtor will execute, but the relevant fn
            // (`type_needs_drop`) is currently factored into
            // `librustc_trans`, so we can't easily do so.
            true
        } else {
            // if type implements Copy, cannot require drop
            false
        }
    }

    fn span_bug(&mut self, span: Self::Span, message: &str) -> ! {
        self.tcx.sess.span_bug(span, message)
    }
}

impl<'a,'tcx:'a> Cx<'a,'tcx> {
    fn cmp_method_ref(&mut self,
                      trait_def_id: DefId,
                      method_name: &str,
                      arg_ty: Ty<'tcx>)
                      -> ItemRef<Cx<'a,'tcx>> {
        let method_name = token::intern(method_name);
        let substs = Substs::new_trait(vec![arg_ty], vec![], arg_ty);
        for trait_item in self.tcx.trait_items(trait_def_id).iter() {
            match *trait_item {
                ty::ImplOrTraitItem::MethodTraitItem(ref method) => {
                    if method.name == method_name {
                        let method_ty = self.tcx.lookup_item_type(method.def_id);
                        let method_ty = method_ty.ty.subst(self.tcx, &substs);
                        return ItemRef {
                            ty: method_ty,
                            def_id: method.def_id,
                            substs: self.tcx.mk_substs(substs),
                        };
                    }
                }
                ty::ImplOrTraitItem::ConstTraitItem(..) |
                ty::ImplOrTraitItem::TypeTraitItem(..) => {
                }
            }
        }

        self.tcx.sess.bug(
            &format!("found no method `{}` in `{:?}`", method_name, trait_def_id));
    }
}

// We only need this impl so that we do deriving for things that are
// defined relative to the `Hair` trait. See `Hair` trait for more
// details.
impl<'a,'tcx> PartialEq for Cx<'a,'tcx> {
    fn eq(&self, _: &Cx<'a,'tcx>) -> bool {
        panic!("Cx should never ACTUALLY be compared for equality")
    }
}

impl<'a,'tcx> Eq for Cx<'a,'tcx> { }

impl<'a,'tcx> Hash for Cx<'a,'tcx> {
    fn hash<H: Hasher>(&self, _: &mut H) {
        panic!("Cx should never ACTUALLY be hashed")
    }
}

impl<'a,'tcx> Debug for Cx<'a,'tcx> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), Error> {
        write!(fmt, "Tcx")
    }
}

mod block;
mod expr;
mod pattern;
mod to_ref;

