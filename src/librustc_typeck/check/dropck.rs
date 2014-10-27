// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use check::regionck::{self, Rcx};

use middle::infer;
use middle::region;
use middle::ty::{self, Ty};
use util::ppaux::{Repr};

use syntax::codemap::Span;

pub fn check_safety_of_destructor_if_necessary<'a, 'tcx>(rcx: &mut Rcx<'a, 'tcx>,
                                                     typ: ty::Ty<'tcx>,
                                                     span: Span,
                                                     scope: region::CodeExtent) {
    debug!("check_safety_of_destructor_if_necessary typ: {} scope: {:?}",
           typ.repr(rcx.tcx()), scope);

    // types that have been traversed so far by `traverse_type_if_unseen`
    let mut breadcrumbs: Vec<Ty<'tcx>> = Vec::new();

    iterate_over_potentially_unsafe_regions_in_type(
        rcx,
        &mut breadcrumbs,
        typ,
        span,
        scope,
        0);
}

fn iterate_over_potentially_unsafe_regions_in_type<'a, 'tcx>(
    rcx: &mut Rcx<'a, 'tcx>,
    breadcrumbs: &mut Vec<Ty<'tcx>>,
    ty_root: ty::Ty<'tcx>,
    span: Span,
    scope: region::CodeExtent,
    depth: uint)
{
    let origin = |&:| infer::SubregionOrigin::SafeDestructor(span);
    let mut walker = ty_root.walk();
    while let Some(typ) = walker.next() {
        // Avoid recursing forever.
        if breadcrumbs.contains(&typ) {
            continue;
        }
        breadcrumbs.push(typ);

        let has_dtor = match typ.sty {
            ty::ty_struct(struct_did, _) => ty::has_dtor(rcx.tcx(), struct_did),
            ty::ty_enum(enum_did, _) => ty::has_dtor(rcx.tcx(), enum_did),
            _ => false,
        };

        debug!("iterate_over_potentially_unsafe_regions_in_type \
                {}typ: {} scope: {:?} has_dtor: {}",
               (0..depth).map(|_| ' ').collect::<String>(),
               typ.repr(rcx.tcx()), scope, has_dtor);

        if has_dtor {
            // If `typ` has a destructor, then we must ensure that all
            // borrowed data reachable via `typ` must outlive the
            // parent of `scope`. (It does not suffice for it to
            // outlive `scope` because that could imply that the
            // borrowed data is torn down in between the end of
            // `scope` and when the destructor itself actually runs.

            let parent_region =
                match rcx.tcx().region_maps.opt_encl_scope(scope) {
                    Some(parent_scope) => ty::ReScope(parent_scope),
                    None => rcx.tcx().sess.span_bug(
                        span, format!("no enclosing scope found for scope: {:?}",
                                      scope).as_slice()),
                };

            regionck::type_must_outlive(rcx, origin(), typ, parent_region);

        } else {
            // Okay, `typ` itself is itself not reachable by a
            // destructor; but it may contain substructure that has a
            // destructor.

            match typ.sty {
                ty::ty_struct(struct_did, substs) => {
                    // Don't recurse; we extract type's substructure,
                    // so do not process subparts of type expression.
                    walker.skip_current_subtree();

                    let fields =
                        ty::lookup_struct_fields(rcx.tcx(), struct_did);
                    for field in fields.iter() {
                        let field_type =
                            ty::lookup_field_type(rcx.tcx(),
                                                  struct_did,
                                                  field.id,
                                                  substs);
                        iterate_over_potentially_unsafe_regions_in_type(
                            rcx,
                            breadcrumbs,
                            field_type,
                            span,
                            scope,
                            depth+1)
                    }
                }

                ty::ty_enum(enum_did, substs) => {
                    // Don't recurse; we extract type's substructure,
                    // so do not process subparts of type expression.
                    walker.skip_current_subtree();

                    let all_variant_info =
                        ty::substd_enum_variants(rcx.tcx(),
                                                 enum_did,
                                                 substs);
                    for variant_info in all_variant_info.iter() {
                        for argument_type in variant_info.args.iter() {
                            iterate_over_potentially_unsafe_regions_in_type(
                                rcx,
                                breadcrumbs,
                                *argument_type,
                                span,
                                scope,
                                depth+1)
                        }
                    }
                }

                ty::ty_rptr(..) | ty::ty_ptr(_) | ty::ty_bare_fn(..) => {
                    // Don't recurse, since references, pointers,
                    // boxes, and bare functions don't own instances
                    // of the types appearing within them.
                    walker.skip_current_subtree();
                }
                _ => {}
            };

            // You might be tempted to pop breadcrumbs here after
            // processing type's internals above, but then you hit
            // exponential time blowup e.g. on
            // compile-fail/huge-struct.rs. Instead, we do not remove
            // anything from the breadcrumbs vector during any particular
            // traversal, and instead clear it after the whole traversal
            // is done.
        }
    }
}
