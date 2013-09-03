// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

# Collect phase

The collect phase of type check has the job of visiting all items,
determining their type, and writing that type into the `tcx.tcache`
table.  Despite its name, this table does not really operate as a
*cache*, at least not for the types of items defined within the
current crate: we assume that after the collect phase, the types of
all local items will be present in the table.

Unlike most of the types that are present in Rust, the types computed
for each item are in fact polytypes.  In "layman's terms", this means
that they are generic types that may have type parameters (more
mathematically phrased, they are universally quantified over a set of
type parameters).  Polytypes are represented by an instance of
`ty::ty_param_bounds_and_ty`.  This combines the core type along with
a list of the bounds for each parameter.  Type parameters themselves
are represented as `ty_param()` instances.

*/


use metadata::csearch;
use middle::ty::{ImplContainer, MethodContainer, TraitContainer, substs};
use middle::ty::{ty_param_bounds_and_ty};
use middle::ty;
use middle::subst::Subst;
use middle::typeck::astconv::{AstConv, ty_of_arg};
use middle::typeck::astconv::{ast_ty_to_ty};
use middle::typeck::astconv;
use middle::typeck::infer;
use middle::typeck::rscope::*;
use middle::typeck::rscope;
use middle::typeck::{CrateCtxt, lookup_def_tcx, no_params, write_ty_to_tcx};
use util::common::pluralize;
use util::ppaux;
use util::ppaux::UserString;

use std::result;
use std::vec;
use syntax::abi::AbiSet;
use syntax::ast::{RegionTyParamBound, TraitTyParamBound};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{local_def, split_trait_methods};
use syntax::codemap::Span;
use syntax::codemap;
use syntax::print::pprust::{path_to_str, explicit_self_to_str};
use syntax::visit;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::parse::token::special_idents;

struct CollectItemTypesVisitor {
    ccx: @mut CrateCtxt
}

impl visit::Visitor<()> for CollectItemTypesVisitor {
    fn visit_item(&mut self, i:@ast::item, _:()) {
        convert(self.ccx, i);
        visit::walk_item(self, i, ());
    }
    fn visit_foreign_item(&mut self, i:@ast::foreign_item, _:()) {
        convert_foreign(self.ccx, i);
        visit::walk_foreign_item(self, i, ());
    }
}

pub fn collect_item_types(ccx: @mut CrateCtxt, crate: &ast::Crate) {
    fn collect_intrinsic_type(ccx: &CrateCtxt,
                              lang_item: ast::DefId) {
        let ty::ty_param_bounds_and_ty { ty: ty, _ } =
            ccx.get_item_ty(lang_item);
        ccx.tcx.intrinsic_defs.insert(lang_item, ty);
    }

    match ccx.tcx.lang_items.ty_desc() {
        Some(id) => { collect_intrinsic_type(ccx, id); } None => {}
    }
    match ccx.tcx.lang_items.opaque() {
        Some(id) => { collect_intrinsic_type(ccx, id); } None => {}
    }

    let mut visitor = CollectItemTypesVisitor{ ccx: ccx };
    visit::walk_crate(&mut visitor, crate, ());
}

pub trait ToTy {
    fn to_ty<RS:RegionScope + Clone + 'static>(
             &self,
             rs: &RS,
             ast_ty: &ast::Ty)
             -> ty::t;
}

impl ToTy for CrateCtxt {
    fn to_ty<RS:RegionScope + Clone + 'static>(
             &self,
             rs: &RS,
             ast_ty: &ast::Ty)
             -> ty::t {
        ast_ty_to_ty(self, rs, ast_ty)
    }
}

impl AstConv for CrateCtxt {
    fn tcx(&self) -> ty::ctxt { self.tcx }

    fn get_item_ty(&self, id: ast::DefId) -> ty::ty_param_bounds_and_ty {
        if id.crate != ast::LOCAL_CRATE {
            csearch::get_type(self.tcx, id)
        } else {
            match self.tcx.items.find(&id.node) {
              Some(&ast_map::node_item(item, _)) => {
                ty_of_item(self, item)
              }
              Some(&ast_map::node_foreign_item(foreign_item, abis, _, _)) => {
                ty_of_foreign_item(self, foreign_item, abis)
              }
              ref x => {
                self.tcx.sess.bug(fmt!("unexpected sort of item \
                                        in get_item_ty(): %?", (*x)));
              }
            }
        }
    }

    fn get_trait_def(&self, id: ast::DefId) -> @ty::TraitDef {
        get_trait_def(self, id)
    }

    fn ty_infer(&self, span: Span) -> ty::t {
        self.tcx.sess.span_bug(span,
                               "found `ty_infer` in unexpected place");
    }
}

pub fn get_enum_variant_types(ccx: &CrateCtxt,
                              enum_ty: ty::t,
                              variants: &[ast::variant],
                              generics: &ast::Generics,
                              rp: Option<ty::region_variance>) {
    let tcx = ccx.tcx;

    // Create a set of parameter types shared among all the variants.
    for variant in variants.iter() {
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);

        // Nullary enum constructors get turned into constants; n-ary enum
        // constructors get turned into functions.
        let result_ty;
        match variant.node.kind {
            ast::tuple_variant_kind(ref args) if args.len() > 0 => {
                let rs = TypeRscope(region_parameterization);
                let input_tys = args.map(|va| ccx.to_ty(&rs, &va.ty));
                result_ty = Some(ty::mk_ctor_fn(tcx, input_tys, enum_ty));
            }

            ast::tuple_variant_kind(_) => {
                result_ty = Some(enum_ty);
            }

            ast::struct_variant_kind(struct_def) => {
                let tpt = ty_param_bounds_and_ty {
                    generics: ty_generics(ccx, rp, generics, 0),
                    ty: enum_ty
                };

                convert_struct(ccx,
                               rp,
                               struct_def,
                               generics,
                               tpt,
                               variant.node.id);

                let input_tys = struct_def.fields.map(
                    |f| ty::node_id_to_type(ccx.tcx, f.node.id));
                result_ty = Some(ty::mk_ctor_fn(tcx, input_tys, enum_ty));
            }
        };

        match result_ty {
            None => {}
            Some(result_ty) => {
                let tpt = ty_param_bounds_and_ty {
                    generics: ty_generics(ccx, rp, generics, 0),
                    ty: result_ty
                };
                tcx.tcache.insert(local_def(variant.node.id), tpt);
                write_ty_to_tcx(tcx, variant.node.id, result_ty);
            }
        }
    }
}

pub fn ensure_trait_methods(ccx: &CrateCtxt,
                            trait_id: ast::NodeId)
{
    let tcx = ccx.tcx;
    let region_paramd = tcx.region_paramd_items.find(&trait_id).map_move(|x| *x);
    match tcx.items.get_copy(&trait_id) {
        ast_map::node_item(@ast::item {
            node: ast::item_trait(ref generics, _, ref ms),
            _
        }, _) => {
            let trait_ty_generics = ty_generics(ccx, region_paramd, generics, 0);

            // For each method, construct a suitable ty::Method and
            // store it into the `tcx.methods` table:
            for m in ms.iter() {
                let ty_method = @match m {
                    &ast::required(ref m) => {
                        ty_method_of_trait_method(
                            ccx, trait_id, region_paramd, generics,
                            &m.id, &m.ident, &m.explicit_self,
                            &m.generics, &m.purity, &m.decl)
                    }

                    &ast::provided(ref m) => {
                        ty_method_of_trait_method(
                            ccx, trait_id, region_paramd, generics,
                            &m.id, &m.ident, &m.explicit_self,
                            &m.generics, &m.purity, &m.decl)
                    }
                };

                if ty_method.explicit_self == ast::sty_static {
                    make_static_method_ty(ccx, trait_id, ty_method,
                                          &trait_ty_generics);
                }

                tcx.methods.insert(ty_method.def_id, ty_method);
            }

            // Add an entry mapping
            let method_def_ids = @ms.map(|m| {
                match m {
                    &ast::required(ref ty_method) => local_def(ty_method.id),
                    &ast::provided(ref method) => local_def(method.id)
                }
            });

            let trait_def_id = local_def(trait_id);
            tcx.trait_method_def_ids.insert(trait_def_id, method_def_ids);
        }
        _ => { /* Ignore things that aren't traits */ }
    }

    fn make_static_method_ty(ccx: &CrateCtxt,
                             trait_id: ast::NodeId,
                             m: &ty::Method,
                             trait_ty_generics: &ty::Generics) {
        // If declaration is
        //
        //     trait<A,B,C> {
        //        fn foo<D,E,F>(...) -> Self;
        //     }
        //
        // and we will create a function like
        //
        //     fn foo<A',B',C',D',E',F',G'>(...) -> D' {}
        //
        // Note that `Self` is replaced with an explicit type
        // parameter D' that is sandwiched in between the trait params
        // and the method params, and thus the indices of the method
        // type parameters are offset by 1 (that is, the method
        // parameters are mapped from D, E, F to E', F', and G').  The
        // choice of this ordering is somewhat arbitrary.
        //
        // Also, this system is rather a hack that should be replaced
        // with a more uniform treatment of Self (which is partly
        // underway).

        // build up a subst that shifts all of the parameters over
        // by one and substitute in a new type param for self

        let tcx = ccx.tcx;

        let dummy_defid = ast::DefId {crate: 0, node: 0};

        // Represents [A',B',C']
        let num_trait_bounds = trait_ty_generics.type_param_defs.len();
        let non_shifted_trait_tps = do vec::from_fn(num_trait_bounds) |i| {
            ty::mk_param(tcx, i, trait_ty_generics.type_param_defs[i].def_id)
        };

        // Represents [D']
        let self_param = ty::mk_param(tcx, num_trait_bounds,
                                      dummy_defid);

        // Represents [E',F',G']
        let num_method_bounds = m.generics.type_param_defs.len();
        let shifted_method_tps = do vec::from_fn(num_method_bounds) |i| {
            ty::mk_param(tcx, i + num_trait_bounds + 1,
                         m.generics.type_param_defs[i].def_id)
        };

        // build up the substitution from
        //     A,B,C => A',B',C'
        //     Self => D'
        //     D,E,F => E',F',G'
        let substs = substs {
            regions: ty::NonerasedRegions(opt_vec::Empty),
            self_ty: Some(self_param),
            tps: non_shifted_trait_tps + shifted_method_tps
        };

        // create the type of `foo`, applying the substitution above
        let ty = ty::subst(tcx,
                           &substs,
                           ty::mk_bare_fn(tcx, m.fty.clone()));

        // create the type parameter definitions for `foo`, applying
        // the substitution to any traits that appear in their bounds.

        // add in the type parameters from the trait
        let mut new_type_param_defs = ~[];
        let substd_type_param_defs =
            trait_ty_generics.type_param_defs.subst(tcx, &substs);
        new_type_param_defs.push_all(*substd_type_param_defs);

        // add in the "self" type parameter
        let self_trait_def = get_trait_def(ccx, local_def(trait_id));
        let self_trait_ref = self_trait_def.trait_ref.subst(tcx, &substs);
        new_type_param_defs.push(ty::TypeParameterDef {
            ident: special_idents::self_,
            def_id: dummy_defid,
            bounds: @ty::ParamBounds {
                builtin_bounds: ty::EmptyBuiltinBounds(),
                trait_bounds: ~[self_trait_ref]
            }
        });

        // add in the type parameters from the method
        let substd_type_param_defs = m.generics.type_param_defs.subst(tcx, &substs);
        new_type_param_defs.push_all(*substd_type_param_defs);

        debug!("static method %s type_param_defs=%s ty=%s, substs=%s",
               m.def_id.repr(tcx),
               new_type_param_defs.repr(tcx),
               ty.repr(tcx),
               substs.repr(tcx));

        tcx.tcache.insert(m.def_id,
                          ty_param_bounds_and_ty {
                              generics: ty::Generics {
                                  type_param_defs: @new_type_param_defs,
                                  region_param: trait_ty_generics.region_param
                              },
                              ty: ty
                          });
    }

    fn ty_method_of_trait_method(this: &CrateCtxt,
                                 trait_id: ast::NodeId,
                                 trait_rp: Option<ty::region_variance>,
                                 trait_generics: &ast::Generics,
                                 m_id: &ast::NodeId,
                                 m_ident: &ast::Ident,
                                 m_explicit_self: &ast::explicit_self,
                                 m_generics: &ast::Generics,
                                 m_purity: &ast::purity,
                                 m_decl: &ast::fn_decl) -> ty::Method
    {
        let trait_self_ty = ty::mk_self(this.tcx, local_def(trait_id));
        let rscope = MethodRscope::new(m_explicit_self.node, trait_rp, trait_generics);
        let (transformed_self_ty, fty) =
            astconv::ty_of_method(this, &rscope, *m_purity, &m_generics.lifetimes,
                                  trait_self_ty, *m_explicit_self, m_decl);
        let num_trait_type_params = trait_generics.ty_params.len();
        ty::Method::new(
            *m_ident,
            ty_generics(this, None, m_generics, num_trait_type_params),
            transformed_self_ty,
            fty,
            m_explicit_self.node,
            // assume public, because this is only invoked on trait methods
            ast::public,
            local_def(*m_id),
            TraitContainer(local_def(trait_id)),
            None
        )
    }
}

pub fn ensure_supertraits(ccx: &CrateCtxt,
                          id: ast::NodeId,
                          sp: codemap::Span,
                          rp: Option<ty::region_variance>,
                          ast_trait_refs: &[ast::trait_ref],
                          generics: &ast::Generics) -> ty::BuiltinBounds
{
    let tcx = ccx.tcx;

    // Called only the first time trait_def_of_item is called.
    // Supertraits are ensured at the same time.
    assert!(!tcx.supertraits.contains_key(&local_def(id)));

    let self_ty = ty::mk_self(ccx.tcx, local_def(id));
    let mut ty_trait_refs: ~[@ty::TraitRef] = ~[];
    let mut bounds = ty::EmptyBuiltinBounds();
    for ast_trait_ref in ast_trait_refs.iter() {
        let trait_def_id = ty::trait_ref_to_def_id(ccx.tcx, ast_trait_ref);
        // FIXME(#8559): Need to instantiate the trait_ref whether or not it's a
        // builtin trait, so that the trait's node id appears in the tcx trait_ref
        // map. This is only needed for metadata; see the similar fixme in encoder.rs.
        let trait_ref = instantiate_trait_ref(ccx, ast_trait_ref, rp,
                                              generics, self_ty);
        if !ty::try_add_builtin_trait(ccx.tcx, trait_def_id, &mut bounds) {

            // FIXME(#5527) Could have same trait multiple times
            if ty_trait_refs.iter().any(|other_trait| other_trait.def_id == trait_ref.def_id) {
                // This means a trait inherited from the same supertrait more
                // than once.
                tcx.sess.span_err(sp, "Duplicate supertrait in trait declaration");
                break;
            } else {
                ty_trait_refs.push(trait_ref);
            }
        }
    }
    tcx.supertraits.insert(local_def(id), @ty_trait_refs);
    bounds
}

/**
 * Checks that a method from an impl/class conforms to the signature of
 * the same method as declared in the trait.
 *
 * # Parameters
 *
 * - impl_tps: the type params declared on the impl itself (not the method!)
 * - cm: info about the method we are checking
 * - trait_m: the method in the trait
 * - trait_substs: the substitutions used on the type of the trait
 * - self_ty: the self type of the impl
 */
pub fn compare_impl_method(tcx: ty::ctxt,
                           impl_tps: uint,
                           cm: &ConvertedMethod,
                           trait_m: &ty::Method,
                           trait_substs: &ty::substs,
                           self_ty: ty::t) {
    debug!("compare_impl_method()");
    let infcx = infer::new_infer_ctxt(tcx);

    let impl_m = &cm.mty;

    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.
    match (&trait_m.explicit_self, &impl_m.explicit_self) {
        (&ast::sty_static, &ast::sty_static) => {}
        (&ast::sty_static, _) => {
            tcx.sess.span_err(
                cm.span,
                fmt!("method `%s` has a `%s` declaration in the impl, \
                      but not in the trait",
                     tcx.sess.str_of(trait_m.ident),
                     explicit_self_to_str(&impl_m.explicit_self, tcx.sess.intr())));
            return;
        }
        (_, &ast::sty_static) => {
            tcx.sess.span_err(
                cm.span,
                fmt!("method `%s` has a `%s` declaration in the trait, \
                      but not in the impl",
                     tcx.sess.str_of(trait_m.ident),
                     explicit_self_to_str(&trait_m.explicit_self, tcx.sess.intr())));
            return;
        }
        _ => {
            // Let the type checker catch other errors below
        }
    }

    let num_impl_m_type_params = impl_m.generics.type_param_defs.len();
    let num_trait_m_type_params = trait_m.generics.type_param_defs.len();
    if num_impl_m_type_params != num_trait_m_type_params {
        tcx.sess.span_err(
            cm.span,
            fmt!("method `%s` has %u type %s, but its trait \
                  declaration has %u type %s",
                 tcx.sess.str_of(trait_m.ident),
                 num_impl_m_type_params,
                 pluralize(num_impl_m_type_params, ~"parameter"),
                 num_trait_m_type_params,
                 pluralize(num_trait_m_type_params, ~"parameter")));
        return;
    }

    if impl_m.fty.sig.inputs.len() != trait_m.fty.sig.inputs.len() {
        tcx.sess.span_err(
            cm.span,
            fmt!("method `%s` has %u parameter%s \
                  but the trait has %u",
                 tcx.sess.str_of(trait_m.ident),
                 impl_m.fty.sig.inputs.len(),
                 if impl_m.fty.sig.inputs.len() == 1 { "" } else { "s" },
                 trait_m.fty.sig.inputs.len()));
        return;
    }

    for (i, trait_param_def) in trait_m.generics.type_param_defs.iter().enumerate() {
        // For each of the corresponding impl ty param's bounds...
        let impl_param_def = &impl_m.generics.type_param_defs[i];

        // Check that the impl does not require any builtin-bounds
        // that the trait does not guarantee:
        let extra_bounds =
            impl_param_def.bounds.builtin_bounds -
            trait_param_def.bounds.builtin_bounds;
        if !extra_bounds.is_empty() {
           tcx.sess.span_err(
               cm.span,
               fmt!("in method `%s`, \
                     type parameter %u requires `%s`, \
                     which is not required by \
                     the corresponding type parameter \
                     in the trait declaration",
                    tcx.sess.str_of(trait_m.ident),
                    i,
                    extra_bounds.user_string(tcx)));
           return;
        }

        // FIXME(#2687)---we should be checking that the bounds of the
        // trait imply the bounds of the subtype, but it appears we
        // are...not checking this.
        if impl_param_def.bounds.trait_bounds.len() !=
            trait_param_def.bounds.trait_bounds.len()
        {
            tcx.sess.span_err(
                cm.span,
                fmt!("in method `%s`, \
                      type parameter %u has %u trait %s, but the \
                      corresponding type parameter in \
                      the trait declaration has %u trait %s",
                     tcx.sess.str_of(trait_m.ident),
                     i, impl_param_def.bounds.trait_bounds.len(),
                     pluralize(impl_param_def.bounds.trait_bounds.len(),
                               ~"bound"),
                     trait_param_def.bounds.trait_bounds.len(),
                     pluralize(trait_param_def.bounds.trait_bounds.len(),
                               ~"bound")));
            return;
        }
    }

    // Replace any references to the self region in the self type with
    // a free region.  So, for example, if the impl type is
    // "&'self str", then this would replace the self type with a free
    // region `self`.
    let dummy_self_r = ty::re_free(ty::FreeRegion {scope_id: cm.body_id,
                                                   bound_region: ty::br_self});
    let self_ty = replace_bound_self(tcx, self_ty, dummy_self_r);

    // We are going to create a synthetic fn type that includes
    // both the method's self argument and its normal arguments.
    // So a method like `fn(&self, a: uint)` would be converted
    // into a function `fn(self: &T, a: uint)`.
    let mut trait_fn_args = ~[];
    let mut impl_fn_args = ~[];

    // For both the trait and the impl, create an argument to
    // represent the self argument (unless this is a static method).
    // This argument will have the *transformed* self type.
    for &t in trait_m.transformed_self_ty.iter() {
        trait_fn_args.push(t);
    }
    for &t in impl_m.transformed_self_ty.iter() {
        impl_fn_args.push(t);
    }

    // Add in the normal arguments.
    trait_fn_args.push_all(trait_m.fty.sig.inputs);
    impl_fn_args.push_all(impl_m.fty.sig.inputs);

    // Create a bare fn type for trait/impl that includes self argument
    let trait_fty =
        ty::mk_bare_fn(tcx,
                       ty::BareFnTy {
                            purity: trait_m.fty.purity,
                            abis: trait_m.fty.abis,
                            sig: ty::FnSig {
                                bound_lifetime_names:
                                    trait_m.fty
                                           .sig
                                           .bound_lifetime_names
                                           .clone(),
                                inputs: trait_fn_args,
                                output: trait_m.fty.sig.output
                            }
                        });
    let impl_fty =
        ty::mk_bare_fn(tcx,
                       ty::BareFnTy {
                            purity: impl_m.fty.purity,
                            abis: impl_m.fty.abis,
                            sig: ty::FnSig {
                                bound_lifetime_names:
                                    impl_m.fty
                                          .sig
                                          .bound_lifetime_names
                                          .clone(),
                                    inputs: impl_fn_args,
                                    output: impl_m.fty.sig.output
                            }
                        });

    // Perform substitutions so that the trait/impl methods are expressed
    // in terms of the same set of type/region parameters:
    // - replace trait type parameters with those from `trait_substs`,
    //   except with any reference to bound self replaced with `dummy_self_r`
    // - replace method parameters on the trait with fresh, dummy parameters
    //   that correspond to the parameters we will find on the impl
    // - replace self region with a fresh, dummy region
    let impl_fty = {
        debug!("impl_fty (pre-subst): %s", ppaux::ty_to_str(tcx, impl_fty));
        replace_bound_self(tcx, impl_fty, dummy_self_r)
    };
    debug!("impl_fty (post-subst): %s", ppaux::ty_to_str(tcx, impl_fty));
    let trait_fty = {
        let num_trait_m_type_params = trait_m.generics.type_param_defs.len();
        let dummy_tps = do vec::from_fn(num_trait_m_type_params) |i| {
            ty::mk_param(tcx, i + impl_tps,
                         impl_m.generics.type_param_defs[i].def_id)
        };
        let trait_tps = trait_substs.tps.map(
            |t| replace_bound_self(tcx, *t, dummy_self_r));
        let substs = substs {
            regions: ty::NonerasedRegions(opt_vec::with(dummy_self_r)),
            self_ty: Some(self_ty),
            tps: vec::append(trait_tps, dummy_tps)
        };
        debug!("trait_fty (pre-subst): %s substs=%s",
               trait_fty.repr(tcx), substs.repr(tcx));
        ty::subst(tcx, &substs, trait_fty)
    };
    debug!("trait_fty (post-subst): %s", trait_fty.repr(tcx));

    match infer::mk_subty(infcx, false, infer::MethodCompatCheck(cm.span),
                          impl_fty, trait_fty) {
        result::Ok(()) => {}
        result::Err(ref terr) => {
            tcx.sess.span_err(
                cm.span,
                fmt!("method `%s` has an incompatible type: %s",
                     tcx.sess.str_of(trait_m.ident),
                     ty::type_err_to_str(tcx, terr)));
            ty::note_and_explain_type_err(tcx, terr);
        }
    }
    return;

    // Replaces bound references to the self region with `with_r`.
    fn replace_bound_self(tcx: ty::ctxt, ty: ty::t,
                          with_r: ty::Region) -> ty::t {
        do ty::fold_regions(tcx, ty) |r, _in_fn| {
            if r == ty::re_bound(ty::br_self) {with_r} else {r}
        }
    }
}

pub fn check_methods_against_trait(ccx: &CrateCtxt,
                                   generics: &ast::Generics,
                                   rp: Option<ty::region_variance>,
                                   selfty: ty::t,
                                   a_trait_ty: &ast::trait_ref,
                                   impl_ms: &[ConvertedMethod])
{
    let tcx = ccx.tcx;
    let trait_ref = instantiate_trait_ref(ccx, a_trait_ty, rp,
                                          generics, selfty);

    if trait_ref.def_id.crate == ast::LOCAL_CRATE {
        ensure_trait_methods(ccx, trait_ref.def_id.node);
    }

    // Check that each method we impl is a method on the trait
    // Trait methods we don't implement must be default methods, but if not
    // we'll catch it in coherence
    let trait_ms = ty::trait_methods(tcx, trait_ref.def_id);
    for impl_m in impl_ms.iter() {
        match trait_ms.iter().find(|trait_m| trait_m.ident == impl_m.mty.ident) {
            Some(trait_m) => {
                let num_impl_tps = generics.ty_params.len();
                compare_impl_method(
                    ccx.tcx, num_impl_tps, impl_m, *trait_m,
                    &trait_ref.substs, selfty);
            }
            None => {
                // This method is not part of the trait
                tcx.sess.span_err(
                    impl_m.span,
                    fmt!("method `%s` is not a member of trait `%s`",
                         tcx.sess.str_of(impl_m.mty.ident),
                         path_to_str(&a_trait_ty.path, tcx.sess.intr())));
            }
        }
    }
} // fn

pub fn convert_field(ccx: &CrateCtxt,
                     rp: Option<ty::region_variance>,
                     type_param_defs: @~[ty::TypeParameterDef],
                     v: &ast::struct_field,
                     generics: &ast::Generics) {
    let region_parameterization =
        RegionParameterization::from_variance_and_generics(rp, generics);
    let tt = ccx.to_ty(&TypeRscope(region_parameterization), &v.node.ty);
    write_ty_to_tcx(ccx.tcx, v.node.id, tt);
    /* add the field to the tcache */
    ccx.tcx.tcache.insert(local_def(v.node.id),
                          ty::ty_param_bounds_and_ty {
                              generics: ty::Generics {
                                  type_param_defs: type_param_defs,
                                  region_param: rp
                              },
                              ty: tt
                          });
}

pub struct ConvertedMethod {
    mty: @ty::Method,
    id: ast::NodeId,
    span: Span,
    body_id: ast::NodeId
}

pub fn convert_methods(ccx: &CrateCtxt,
                       container: MethodContainer,
                       ms: &[@ast::method],
                       untransformed_rcvr_ty: ty::t,
                       rcvr_ty_generics: &ty::Generics,
                       rcvr_ast_generics: &ast::Generics,
                       rcvr_visibility: ast::visibility)
                    -> ~[ConvertedMethod]
{
    let tcx = ccx.tcx;
    return ms.iter().map(|m| {
        let num_rcvr_ty_params = rcvr_ty_generics.type_param_defs.len();
        let m_ty_generics =
            ty_generics(ccx, rcvr_ty_generics.region_param, &m.generics,
                        num_rcvr_ty_params);
        let mty = @ty_of_method(ccx,
                                container,
                                *m,
                                rcvr_ty_generics.region_param,
                                untransformed_rcvr_ty,
                                rcvr_ast_generics,
                                rcvr_visibility,
                                &m.generics);
        let fty = ty::mk_bare_fn(tcx, mty.fty.clone());
        tcx.tcache.insert(
            local_def(m.id),

            // n.b.: the type of a method is parameterized by both
            // the tps on the receiver and those on the method itself
            ty_param_bounds_and_ty {
                generics: ty::Generics {
                    type_param_defs: @vec::append(
                        (*rcvr_ty_generics.type_param_defs).clone(),
                        *m_ty_generics.type_param_defs),
                    region_param: rcvr_ty_generics.region_param
                },
                ty: fty
            });
        write_ty_to_tcx(tcx, m.id, fty);
        tcx.methods.insert(mty.def_id, mty);
        ConvertedMethod {mty: mty, id: m.id,
                         span: m.span, body_id: m.body.id}
    }).collect();

    fn ty_of_method(ccx: &CrateCtxt,
                    container: MethodContainer,
                    m: &ast::method,
                    rp: Option<ty::region_variance>,
                    untransformed_rcvr_ty: ty::t,
                    rcvr_generics: &ast::Generics,
                    rcvr_visibility: ast::visibility,
                    method_generics: &ast::Generics) -> ty::Method
    {
        let rscope = MethodRscope::new(m.explicit_self.node,
                                       rp,
                                       rcvr_generics);
        let (transformed_self_ty, fty) =
            astconv::ty_of_method(ccx, &rscope, m.purity,
                                  &method_generics.lifetimes,
                                  untransformed_rcvr_ty,
                                  m.explicit_self, &m.decl);

        // if the method specifies a visibility, use that, otherwise
        // inherit the visibility from the impl (so `foo` in `pub impl
        // { fn foo(); }` is public, but private in `priv impl { fn
        // foo(); }`).
        let method_vis = m.vis.inherit_from(rcvr_visibility);

        let num_rcvr_type_params = rcvr_generics.ty_params.len();
        ty::Method::new(
            m.ident,
            ty_generics(ccx, None, &m.generics, num_rcvr_type_params),
            transformed_self_ty,
            fty,
            m.explicit_self.node,
            method_vis,
            local_def(m.id),
            container,
            None
        )
    }
}

pub fn ensure_no_ty_param_bounds(ccx: &CrateCtxt,
                                 span: Span,
                                 generics: &ast::Generics,
                                 thing: &'static str) {
    for ty_param in generics.ty_params.iter() {
        if ty_param.bounds.len() > 0 {
            ccx.tcx.sess.span_err(
                span,
                fmt!("trait bounds are not allowed in %s definitions",
                     thing));
        }
    }
}

pub fn convert(ccx: &CrateCtxt, it: &ast::item) {
    let tcx = ccx.tcx;
    let rp = tcx.region_paramd_items.find(&it.id).map_move(|x| *x);
    debug!("convert: item %s with id %d rp %?",
           tcx.sess.str_of(it.ident), it.id, rp);
    match it.node {
      // These don't define types.
      ast::item_foreign_mod(_) | ast::item_mod(_) => {}
      ast::item_enum(ref enum_definition, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "enumeration");
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        get_enum_variant_types(ccx,
                               tpt.ty,
                               enum_definition.variants,
                               generics,
                               rp);
      }
      ast::item_impl(ref generics, ref opt_trait_ref, ref selfty, ref ms) => {
        let i_ty_generics = ty_generics(ccx, rp, generics, 0);
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);
        let selfty = ccx.to_ty(&TypeRscope(region_parameterization), selfty);
        write_ty_to_tcx(tcx, it.id, selfty);
        tcx.tcache.insert(local_def(it.id),
                          ty_param_bounds_and_ty {
                              generics: i_ty_generics,
                              ty: selfty});

        // If there is a trait reference, treat the methods as always public.
        // This is to work around some incorrect behavior in privacy checking:
        // when the method belongs to a trait, it should acquire the privacy
        // from the trait, not the impl. Forcing the visibility to be public
        // makes things sorta work.
        let parent_visibility = if opt_trait_ref.is_some() {
            ast::public
        } else {
            it.vis
        };

        let cms = convert_methods(ccx,
                                  ImplContainer(local_def(it.id)),
                                  *ms,
                                  selfty,
                                  &i_ty_generics,
                                  generics,
                                  parent_visibility);
        for t in opt_trait_ref.iter() {
            // Prevent the builtin kind traits from being manually implemented.
            let trait_def_id = ty::trait_ref_to_def_id(tcx, t);
            if tcx.lang_items.to_builtin_kind(trait_def_id).is_some() {
                tcx.sess.span_err(it.span,
                    "cannot provide an explicit implementation \
                     for a builtin kind");
            }

            check_methods_against_trait(ccx, generics, rp, selfty, t, cms);
        }
      }
      ast::item_trait(ref generics, _, ref trait_methods) => {
          let _trait_def = trait_def_of_item(ccx, it);

          // Run convert_methods on the provided methods.
          let (_, provided_methods) =
              split_trait_methods(*trait_methods);
          let untransformed_rcvr_ty = ty::mk_self(tcx, local_def(it.id));
          let (ty_generics, _) = mk_item_substs(ccx, generics, rp,
                                                Some(untransformed_rcvr_ty));
          let _ = convert_methods(ccx,
                                  TraitContainer(local_def(it.id)),
                                  provided_methods,
                                  untransformed_rcvr_ty,
                                  &ty_generics,
                                  generics,
                                  it.vis);

          // We need to do this *after* converting methods, since
          // convert_methods produces a tcache entry that is wrong for
          // static trait methods. This is somewhat unfortunate.
          ensure_trait_methods(ccx, it.id);
      }
      ast::item_struct(struct_def, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "structure");

        // Write the class type
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
        tcx.tcache.insert(local_def(it.id), tpt);

        convert_struct(ccx, rp, struct_def, generics, tpt, it.id);
      }
      ast::item_ty(_, ref generics) => {
        ensure_no_ty_param_bounds(ccx, it.span, generics, "type");
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
      }
      _ => {
        // This call populates the type cache with the converted type
        // of the item in passing. All we have to do here is to write
        // it into the node type table.
        let tpt = ty_of_item(ccx, it);
        write_ty_to_tcx(tcx, it.id, tpt.ty);
      }
    }
}

pub fn convert_struct(ccx: &CrateCtxt,
                      rp: Option<ty::region_variance>,
                      struct_def: &ast::struct_def,
                      generics: &ast::Generics,
                      tpt: ty::ty_param_bounds_and_ty,
                      id: ast::NodeId) {
    let tcx = ccx.tcx;

    // Write the type of each of the members
    for f in struct_def.fields.iter() {
       convert_field(ccx, rp, tpt.generics.type_param_defs, *f, generics);
    }
    let (_, substs) = mk_item_substs(ccx, generics, rp, None);
    let selfty = ty::mk_struct(tcx, local_def(id), substs);

    // If this struct is enum-like or tuple-like, create the type of its
    // constructor.
    match struct_def.ctor_id {
        None => {}
        Some(ctor_id) => {
            if struct_def.fields.len() == 0 {
                // Enum-like.
                write_ty_to_tcx(tcx, ctor_id, selfty);
                tcx.tcache.insert(local_def(ctor_id), tpt);
            } else if struct_def.fields[0].node.kind == ast::unnamed_field {
                // Tuple-like.
                let inputs =
                    struct_def.fields.map(
                        |field| ccx.tcx.tcache.get(
                            &local_def(field.node.id)).ty);
                let ctor_fn_ty = ty::mk_ctor_fn(tcx, inputs, selfty);
                write_ty_to_tcx(tcx, ctor_id, ctor_fn_ty);
                tcx.tcache.insert(local_def(ctor_id), ty_param_bounds_and_ty {
                    generics: tpt.generics,
                    ty: ctor_fn_ty
                });
            }
        }
    }
}

pub fn convert_foreign(ccx: &CrateCtxt, i: &ast::foreign_item) {
    // As above, this call populates the type table with the converted
    // type of the foreign item. We simply write it into the node type
    // table.

    // For reasons I cannot fully articulate, I do so hate the AST
    // map, and I regard each time that I use it as a personal and
    // moral failing, but at the moment it seems like the only
    // convenient way to extract the ABI. - ndm
    let abis = match ccx.tcx.items.find(&i.id) {
        Some(&ast_map::node_foreign_item(_, abis, _, _)) => abis,
        ref x => {
            ccx.tcx.sess.bug(fmt!("unexpected sort of item \
                                   in get_item_ty(): %?", (*x)));
        }
    };

    let tpt = ty_of_foreign_item(ccx, i, abis);
    write_ty_to_tcx(ccx.tcx, i.id, tpt.ty);
    ccx.tcx.tcache.insert(local_def(i.id), tpt);
}

pub fn instantiate_trait_ref(ccx: &CrateCtxt,
                             ast_trait_ref: &ast::trait_ref,
                             rp: Option<ty::region_variance>,
                             generics: &ast::Generics,
                             self_ty: ty::t) -> @ty::TraitRef
{
    /*!
     * Instantiates the path for the given trait reference, assuming that
     * it's bound to a valid trait type. Returns the def_id for the defining
     * trait. Fails if the type is a type other than an trait type.
     */

    let rp = RegionParameterization::from_variance_and_generics(rp, generics);

    let rscope = TypeRscope(rp);

    match lookup_def_tcx(ccx.tcx, ast_trait_ref.path.span, ast_trait_ref.ref_id) {
        ast::DefTrait(trait_did) => {
            let trait_ref =
                astconv::ast_path_to_trait_ref(
                    ccx, &rscope, trait_did, Some(self_ty), &ast_trait_ref.path);
            ccx.tcx.trait_refs.insert(
                ast_trait_ref.ref_id, trait_ref);
            return trait_ref;
        }
        _ => {
            ccx.tcx.sess.span_fatal(
                ast_trait_ref.path.span,
                fmt!("%s is not a trait",
                    path_to_str(&ast_trait_ref.path,
                                ccx.tcx.sess.intr())));
        }
    }
}

fn get_trait_def(ccx: &CrateCtxt, trait_id: ast::DefId) -> @ty::TraitDef {
    if trait_id.crate != ast::LOCAL_CRATE {
        ty::lookup_trait_def(ccx.tcx, trait_id)
    } else {
        match ccx.tcx.items.get(&trait_id.node) {
            &ast_map::node_item(item, _) => trait_def_of_item(ccx, item),
            _ => ccx.tcx.sess.bug(fmt!("get_trait_def(%d): not an item",
                                       trait_id.node))
        }
    }
}

pub fn trait_def_of_item(ccx: &CrateCtxt, it: &ast::item) -> @ty::TraitDef {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    match tcx.trait_defs.find(&def_id) {
      Some(&def) => return def,
      _ => {}
    }
    let rp = tcx.region_paramd_items.find(&it.id).map_move(|x| *x);
    match it.node {
        ast::item_trait(ref generics, ref supertraits, _) => {
            let self_ty = ty::mk_self(tcx, def_id);
            let (ty_generics, substs) = mk_item_substs(ccx, generics, rp,
                                                       Some(self_ty));
            let bounds = ensure_supertraits(ccx, it.id, it.span, rp,
                                            *supertraits, generics);
            let trait_ref = @ty::TraitRef {def_id: def_id,
                                           substs: substs};
            let trait_def = @ty::TraitDef {generics: ty_generics,
                                           bounds: bounds,
                                           trait_ref: trait_ref};
            tcx.trait_defs.insert(def_id, trait_def);
            return trait_def;
        }
        ref s => {
            tcx.sess.span_bug(
                it.span,
                fmt!("trait_def_of_item invoked on %?", s));
        }
    }
}

pub fn ty_of_item(ccx: &CrateCtxt, it: &ast::item)
               -> ty::ty_param_bounds_and_ty {
    let def_id = local_def(it.id);
    let tcx = ccx.tcx;
    match tcx.tcache.find(&def_id) {
      Some(&tpt) => return tpt,
      _ => {}
    }
    let rp = tcx.region_paramd_items.find(&it.id).map_move(|x| *x);
    match it.node {
      ast::item_static(ref t, _, _) => {
        let typ = ccx.to_ty(&EmptyRscope, t);
        let tpt = no_params(typ);
        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_fn(ref decl, purity, abi, ref generics, _) => {
        assert!(rp.is_none());
        let ty_generics = ty_generics(ccx, None, generics, 0);
        let tofd = astconv::ty_of_bare_fn(ccx,
                                          &EmptyRscope,
                                          purity,
                                          abi,
                                          &generics.lifetimes,
                                          decl);
        let tpt = ty_param_bounds_and_ty {
            generics: ty::Generics {
                type_param_defs: ty_generics.type_param_defs,
                region_param: None
            },
            ty: ty::mk_bare_fn(ccx.tcx, tofd)
        };
        debug!("type of %s (id %d) is %s",
               tcx.sess.str_of(it.ident),
               it.id,
               ppaux::ty_to_str(tcx, tpt.ty));
        ccx.tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_ty(ref t, ref generics) => {
        match tcx.tcache.find(&local_def(it.id)) {
          Some(&tpt) => return tpt,
          None => { }
        }

        let rp = tcx.region_paramd_items.find(&it.id).map_move(|x| *x);
        let region_parameterization =
            RegionParameterization::from_variance_and_generics(rp, generics);
        let tpt = {
            let ty = ccx.to_ty(&TypeRscope(region_parameterization), t);
            ty_param_bounds_and_ty {
                generics: ty_generics(ccx, rp, generics, 0),
                ty: ty
            }
        };

        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_enum(_, ref generics) => {
        // Create a new generic polytype.
        let (ty_generics, substs) = mk_item_substs(ccx, generics, rp, None);
        let t = ty::mk_enum(tcx, local_def(it.id), substs);
        let tpt = ty_param_bounds_and_ty {
            generics: ty_generics,
            ty: t
        };
        tcx.tcache.insert(local_def(it.id), tpt);
        return tpt;
      }
      ast::item_trait(*) => {
          tcx.sess.span_bug(
              it.span,
              fmt!("Invoked ty_of_item on trait"));
      }
      ast::item_struct(_, ref generics) => {
          let (ty_generics, substs) = mk_item_substs(ccx, generics, rp, None);
          let t = ty::mk_struct(tcx, local_def(it.id), substs);
          let tpt = ty_param_bounds_and_ty {
              generics: ty_generics,
              ty: t
          };
          tcx.tcache.insert(local_def(it.id), tpt);
          return tpt;
      }
      ast::item_impl(*) | ast::item_mod(_) |
      ast::item_foreign_mod(_) => fail!(),
      ast::item_mac(*) => fail!("item macros unimplemented")
    }
}

pub fn ty_of_foreign_item(ccx: &CrateCtxt,
                          it: &ast::foreign_item,
                          abis: AbiSet) -> ty::ty_param_bounds_and_ty
{
    match it.node {
        ast::foreign_item_fn(ref fn_decl, ref generics) => {
            ty_of_foreign_fn_decl(ccx,
                                  fn_decl,
                                  local_def(it.id),
                                  generics,
                                  abis)
        }
        ast::foreign_item_static(ref t, _) => {
            ty::ty_param_bounds_and_ty {
                generics: ty::Generics {
                    type_param_defs: @~[],
                    region_param: None,
                },
                ty: ast_ty_to_ty(ccx, &EmptyRscope, t)
            }
        }
    }
}

pub fn ty_generics(ccx: &CrateCtxt,
                   rp: Option<ty::region_variance>,
                   generics: &ast::Generics,
                   base_index: uint) -> ty::Generics {
    return ty::Generics {
        region_param: rp,
        type_param_defs: @generics.ty_params.mapi_to_vec(|offset, param| {
            match ccx.tcx.ty_param_defs.find(&param.id) {
                Some(&def) => def,
                None => {
                    let param_ty = ty::param_ty {idx: base_index + offset,
                                                 def_id: local_def(param.id)};
                    let bounds = @compute_bounds(ccx, rp, generics,
                                                 param_ty, &param.bounds);
                    let def = ty::TypeParameterDef {
                        ident: param.ident,
                        def_id: local_def(param.id),
                        bounds: bounds
                    };
                    debug!("def for param: %s", def.repr(ccx.tcx));
                    ccx.tcx.ty_param_defs.insert(param.id, def);
                    def
                }
            }
        })
    };

    fn compute_bounds(
        ccx: &CrateCtxt,
        rp: Option<ty::region_variance>,
        generics: &ast::Generics,
        param_ty: ty::param_ty,
        ast_bounds: &OptVec<ast::TyParamBound>) -> ty::ParamBounds
    {
        /*!
         *
         * Translate the AST's notion of ty param bounds (which are an
         * enum consisting of a newtyped Ty or a region) to ty's
         * notion of ty param bounds, which can either be user-defined
         * traits, or one of the two built-in traits (formerly known
         * as kinds): Freeze and Send.
         */

        let mut param_bounds = ty::ParamBounds {
            builtin_bounds: ty::EmptyBuiltinBounds(),
            trait_bounds: ~[]
        };
        for ast_bound in ast_bounds.iter() {
            match *ast_bound {
                TraitTyParamBound(ref b) => {
                    let ty = ty::mk_param(ccx.tcx, param_ty.idx, param_ty.def_id);
                    let trait_ref = instantiate_trait_ref(ccx, b, rp, generics, ty);
                    if !ty::try_add_builtin_trait(
                        ccx.tcx, trait_ref.def_id,
                        &mut param_bounds.builtin_bounds)
                    {
                        // Must be a user-defined trait
                        param_bounds.trait_bounds.push(trait_ref);
                    }
                }

                RegionTyParamBound => {
                    param_bounds.builtin_bounds.add(ty::BoundStatic);
                }
            }
        }

        param_bounds
    }
}

pub fn ty_of_foreign_fn_decl(ccx: &CrateCtxt,
                             decl: &ast::fn_decl,
                             def_id: ast::DefId,
                             ast_generics: &ast::Generics,
                             abis: AbiSet)
                          -> ty::ty_param_bounds_and_ty {
    let ty_generics = ty_generics(ccx, None, ast_generics, 0);
    let region_param_names = RegionParamNames::from_generics(ast_generics);
    let rb = in_binding_rscope(&EmptyRscope, region_param_names);
    let input_tys = decl.inputs.map(|a| ty_of_arg(ccx, &rb, a, None) );
    let output_ty = ast_ty_to_ty(ccx, &rb, &decl.output);

    let t_fn = ty::mk_bare_fn(
        ccx.tcx,
        ty::BareFnTy {
            abis: abis,
            purity: ast::unsafe_fn,
            sig: ty::FnSig {bound_lifetime_names: opt_vec::Empty,
                            inputs: input_tys,
                            output: output_ty}
        });
    let tpt = ty_param_bounds_and_ty {
        generics: ty_generics,
        ty: t_fn
    };
    ccx.tcx.tcache.insert(def_id, tpt);
    return tpt;
}

pub fn mk_item_substs(ccx: &CrateCtxt,
                      ast_generics: &ast::Generics,
                      rp: Option<ty::region_variance>,
                      self_ty: Option<ty::t>) -> (ty::Generics, ty::substs)
{
    let mut i = 0;
    let ty_generics = ty_generics(ccx, rp, ast_generics, 0);
    let params = ast_generics.ty_params.map_to_vec(|atp| {
        let t = ty::mk_param(ccx.tcx, i, local_def(atp.id));
        i += 1u;
        t
    });
    let regions = rscope::bound_self_region(rp);
    (ty_generics, substs {regions: ty::NonerasedRegions(regions),
                          self_ty: self_ty,
                          tps: params})
}
