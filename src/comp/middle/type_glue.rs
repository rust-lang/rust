// The "shape" of a type is best defined as "how a value of a type looks from
// the standpoint of a certain built-in operation".
//
// This is used to collapse take/drop glues that would otherwise be
// separate. For instance, a boxed tuple of 3 ints and a boxed tuple of 3
// uints look the same as far as reference count manipulation is concerned, so
// they get the same shape so that their reference count glues can be
// collapsed together. To give another example, an int and float have the
// same (nonexistent!) glue as far as reference counting is concerned, since
// they aren't reference counted.

import front::ast;
import middle::trans::variant_info;
import middle::ty;

type variant_getter = fn(&ast::def_id) -> vec[variant_info];


// Reference counting shapes.

tag rc_shape {
    rs_none;                        // No reference count.
    rs_ref;                         // Reference counted box.
    rs_pair;                        // Pair of code/const ptr + rc box.
    rs_tag(vec[vec[@rc_shape]]);    // Discriminated union.
    rs_tup(vec[@rc_shape]);         // Tuple.
}

fn rc_shape_of(&ty::ctxt tcx, variant_getter getter, ty::t t) -> rc_shape {
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
            let vec[vec[@rc_shape]] result = [];

            auto vinfos = getter(did);
            for (variant_info vinfo in vinfos) {
                let vec[@rc_shape] variant_rcs = [];
                for (ty::t typ in vinfo.args) {
                    auto ty_1 = ty::bind_params_in_type(tcx, typ);
                    ty_1 = ty::substitute_type_params(tcx, params, ty_1);
                    variant_rcs += [@rc_shape_of(tcx, getter, ty_1)];
                }
                result += [variant_rcs];
            }

            ret rs_tag(result);
        }
        case (ty::ty_box(_)) { ret rs_ref; }
        case (ty::ty_vec(_)) { ret rs_ref; }
        case (ty::ty_port(_)) { ret rs_ref; }
        case (ty::ty_chan(_)) { ret rs_ref; }
        case (ty::ty_task) { ret rs_ref; }
        case (ty::ty_tup(?mts)) {
            let vec[@rc_shape] result = [];
            for (ty::mt tm in mts) {
                result += [@rc_shape_of(tcx, getter, tm.ty)];
            }
            ret rs_tup(result);
        }
        case (ty::ty_rec(?fields)) {
            let vec[@rc_shape] result = [];
            for (ty::field fld in fields) {
                result += [@rc_shape_of(tcx, getter, fld.mt.ty)];
            }
            ret rs_tup(result);
        }
        case (ty::ty_fn(_, _, _)) { ret rs_pair; }
        case (ty::ty_native_fn(_, _, _)) { ret rs_pair; }
        case (ty::ty_obj(_)) { ret rs_pair; }
        case (ty::ty_var(_)) { log_err "var in rc_shape_of()"; fail; }
        case (ty::ty_local(_)) {
            log_err "local in rc_shape_of()";
            fail;
        }
        case (ty::ty_param(_)) {
            log_err "param in rc_shape_of()";
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

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
