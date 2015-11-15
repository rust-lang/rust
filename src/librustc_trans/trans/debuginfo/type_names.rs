// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Type Names for Debug Info.

use super::namespace::crate_root_namespace;

use trans::common::CrateContext;
use middle::def_id::DefId;
use middle::infer;
use middle::subst::{self, Substs};
use middle::ty::{self, Ty};

use rustc_front::hir;
use syntax::ast;

// Compute the name of the type as it should be stored in debuginfo. Does not do
// any caching, i.e. calling the function twice with the same type will also do
// the work twice. The `qualified` parameter only affects the first level of the
// type name, further levels (i.e. type parameters) are always fully qualified.
pub fn compute_debuginfo_type_name<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                             t: Ty<'tcx>,
                                             qualified: bool)
                                             -> String {
    let mut result = String::with_capacity(64);
    push_debuginfo_type_name(cx, t, qualified, &mut result);
    result
}

// Pushes the name of the type as it should be stored in debuginfo on the
// `output` String. See also compute_debuginfo_type_name().
pub fn push_debuginfo_type_name<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                          t: Ty<'tcx>,
                                          qualified: bool,
                                          output: &mut String) {
    match t.sty {
        ty::TyBool              => output.push_str("bool"),
        ty::TyChar              => output.push_str("char"),
        ty::TyStr               => output.push_str("str"),
        ty::TyInt(ast::TyIs)    => output.push_str("isize"),
        ty::TyInt(ast::TyI8)    => output.push_str("i8"),
        ty::TyInt(ast::TyI16)   => output.push_str("i16"),
        ty::TyInt(ast::TyI32)   => output.push_str("i32"),
        ty::TyInt(ast::TyI64)   => output.push_str("i64"),
        ty::TyUint(ast::TyUs)   => output.push_str("usize"),
        ty::TyUint(ast::TyU8)   => output.push_str("u8"),
        ty::TyUint(ast::TyU16)  => output.push_str("u16"),
        ty::TyUint(ast::TyU32)  => output.push_str("u32"),
        ty::TyUint(ast::TyU64)  => output.push_str("u64"),
        ty::TyFloat(ast::TyF32) => output.push_str("f32"),
        ty::TyFloat(ast::TyF64) => output.push_str("f64"),
        ty::TyStruct(def, substs) |
        ty::TyEnum(def, substs) => {
            push_item_name(cx, def.did, qualified, output);
            push_type_params(cx, substs, output);
        },
        ty::TyTuple(ref component_types) => {
            output.push('(');
            for &component_type in component_types {
                push_debuginfo_type_name(cx, component_type, true, output);
                output.push_str(", ");
            }
            if !component_types.is_empty() {
                output.pop();
                output.pop();
            }
            output.push(')');
        },
        ty::TyBox(inner_type) => {
            output.push_str("Box<");
            push_debuginfo_type_name(cx, inner_type, true, output);
            output.push('>');
        },
        ty::TyRawPtr(ty::TypeAndMut { ty: inner_type, mutbl } ) => {
            output.push('*');
            match mutbl {
                hir::MutImmutable => output.push_str("const "),
                hir::MutMutable => output.push_str("mut "),
            }

            push_debuginfo_type_name(cx, inner_type, true, output);
        },
        ty::TyRef(_, ty::TypeAndMut { ty: inner_type, mutbl }) => {
            output.push('&');
            if mutbl == hir::MutMutable {
                output.push_str("mut ");
            }

            push_debuginfo_type_name(cx, inner_type, true, output);
        },
        ty::TyArray(inner_type, len) => {
            output.push('[');
            push_debuginfo_type_name(cx, inner_type, true, output);
            output.push_str(&format!("; {}", len));
            output.push(']');
        },
        ty::TySlice(inner_type) => {
            output.push('[');
            push_debuginfo_type_name(cx, inner_type, true, output);
            output.push(']');
        },
        ty::TyTrait(ref trait_data) => {
            let principal = cx.tcx().erase_late_bound_regions(&trait_data.principal);
            push_item_name(cx, principal.def_id, false, output);
            push_type_params(cx, principal.substs, output);
        },
        ty::TyBareFn(_, &ty::BareFnTy{ unsafety, abi, ref sig } ) => {
            if unsafety == hir::Unsafety::Unsafe {
                output.push_str("unsafe ");
            }

            if abi != ::syntax::abi::Rust {
                output.push_str("extern \"");
                output.push_str(abi.name());
                output.push_str("\" ");
            }

            output.push_str("fn(");

            let sig = cx.tcx().erase_late_bound_regions(sig);
            let sig = infer::normalize_associated_type(cx.tcx(), &sig);
            if !sig.inputs.is_empty() {
                for &parameter_type in &sig.inputs {
                    push_debuginfo_type_name(cx, parameter_type, true, output);
                    output.push_str(", ");
                }
                output.pop();
                output.pop();
            }

            if sig.variadic {
                if !sig.inputs.is_empty() {
                    output.push_str(", ...");
                } else {
                    output.push_str("...");
                }
            }

            output.push(')');

            match sig.output {
                ty::FnConverging(result_type) if result_type.is_nil() => {}
                ty::FnConverging(result_type) => {
                    output.push_str(" -> ");
                    push_debuginfo_type_name(cx, result_type, true, output);
                }
                ty::FnDiverging => {
                    output.push_str(" -> !");
                }
            }
        },
        ty::TyClosure(..) => {
            output.push_str("closure");
        }
        ty::TyError |
        ty::TyInfer(_) |
        ty::TyProjection(..) |
        ty::TyParam(_) => {
            cx.sess().bug(&format!("debuginfo: Trying to create type name for \
                unexpected type: {:?}", t));
        }
    }

    fn push_item_name(cx: &CrateContext,
                      def_id: DefId,
                      qualified: bool,
                      output: &mut String) {
        cx.tcx().with_path(def_id, |path| {
            if qualified {
                if def_id.is_local() {
                    output.push_str(crate_root_namespace(cx));
                    output.push_str("::");
                }

                let mut path_element_count = 0;
                for path_element in path {
                    output.push_str(&path_element.name().as_str());
                    output.push_str("::");
                    path_element_count += 1;
                }

                if path_element_count == 0 {
                    cx.sess().bug("debuginfo: Encountered empty item path!");
                }

                output.pop();
                output.pop();
            } else {
                let name = path.last().expect("debuginfo: Empty item path?").name();
                output.push_str(&name.as_str());
            }
        });
    }

    // Pushes the type parameters in the given `Substs` to the output string.
    // This ignores region parameters, since they can't reliably be
    // reconstructed for items from non-local crates. For local crates, this
    // would be possible but with inlining and LTO we have to use the least
    // common denominator - otherwise we would run into conflicts.
    fn push_type_params<'a, 'tcx>(cx: &CrateContext<'a, 'tcx>,
                                  substs: &subst::Substs<'tcx>,
                                  output: &mut String) {
        if substs.types.is_empty() {
            return;
        }

        output.push('<');

        for &type_parameter in &substs.types {
            push_debuginfo_type_name(cx, type_parameter, true, output);
            output.push_str(", ");
        }

        output.pop();
        output.pop();

        output.push('>');
    }
}
