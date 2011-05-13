// The "shape" of a type is best defined as "how a value of a type looks from
// the standpoint of a certain built-in operation".
//
// This is used to collapse glues that would otherwise be separate. For
// instance, a boxed tuple of 3 ints and a boxed tuple of 3 uints look the
// same as far as reference count manipulation is concerned, so they get the
// same shape so that their reference count glues can be collapsed together.
// To give another example, an int and float have the same (nonexistent!) glue
// as far as reference counting is concerned, since they aren't reference
// counted.

import front::ast;
import middle::trans::variant_info;
import middle::ty;

type variant_getter = fn(&ast::def_id) -> vec[variant_info];


// Reference counting shapes.

mod rc {
    // TODO: Re-export, so that users can just say shape::rc.
    // FIXME: The bottom two should be just "vec[rc]", but that causes an
    // infinite loop in trans.
    tag rc {
        rs_none;                        // No reference count.
        rs_ref;                         // Reference counted box.
        rs_tag(vec[@rc]);               // Discriminated union.
        rs_tup(vec[@rc]);               // Tuple.
    }

    fn shape_of(&ty::ctxt tcx, variant_getter getter, ty::t t) -> rc {
        alt (ty::struct(tcx, t)) {
            // TODO: Or-patterns
            case (ty::ty_nil) { ret rs_none; }
            case (ty::ty_bool) { ret rs_none; }
            case (ty::ty_int) { ret rs_none; }
            case (ty::ty_uint) { ret rs_none; }
            case (ty::ty_machine(_)) { ret rs_none; }
            case (ty::ty_char) { ret rs_none; }
            case (ty::ty_str) { ret rs_none; }
            case (ty::ty_tag(?did, ?params)) {
                let vec[@rc] result = vec();

                auto vinfos = getter(did);
                for (variant_info vinfo in vinfos) {
                    let vec[@rc] variant_rcs = vec();
                    for (ty::t typ in vinfo.args) {
                        auto ty_1 = ty::bind_params_in_type(tcx, typ);
                        ty_1 = ty::substitute_type_params(tcx, params, ty_1);
                        variant_rcs += vec(@shape_of(tcx, getter, ty_1));
                    }
                    result += vec(@rs_tup(variant_rcs));
                }

                ret rs_tag(result);
            }
            case (ty::ty_box(_)) { ret rs_ref; }
            case (ty::ty_vec(_)) { ret rs_ref; }
            case (ty::ty_port(_)) { ret rs_ref; }
            case (ty::ty_chan(_)) { ret rs_ref; }
            case (ty::ty_task) { ret rs_ref; }
            case (ty::ty_tup(?mts)) {
                let vec[@rc] result = vec();
                for (ty::mt tm in mts) {
                    result += vec(@shape_of(tcx, getter, tm.ty));
                }
                ret rs_tup(result);
            }
            case (ty::ty_rec(?fields)) {
                let vec[@rc] result = vec();
                for (ty::field fld in fields) {
                    result += vec(@shape_of(tcx, getter, fld.mt.ty));
                }
                ret rs_tup(result);
            }
            case (ty::ty_fn(_, _, _)) { ret rs_ref; }
            case (ty::ty_native_fn(_, _, _)) { ret rs_ref; }
            case (ty::ty_obj(_)) { ret rs_ref; }
            case (ty::ty_var(_)) { log_err "var in rc::shape_of()"; fail; }
            case (ty::ty_local(_)) {
                log_err "local in rc::shape_of()";
                fail;
            }
            case (ty::ty_param(_)) {
                log_err "param in rc::shape_of()";
                fail;
            }
            case (ty::ty_bound_param(_)) {
                log_err "bound param in rc::shape_of()";
                fail;
            }
            case (ty::ty_type) { ret rs_ref; }
            case (ty::ty_native) { ret rs_none; }
        }
    }
}

