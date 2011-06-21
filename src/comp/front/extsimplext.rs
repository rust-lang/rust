use std;

import util::common::span;
import std::vec;
import std::option;
import vec::map;
import vec::len;
import option::some;
import option::none;

import ext::syntax_extension;
import ext::ext_ctxt;
import ext::normal;
import ext::expr_to_str;
import ext::expr_to_ident;

import fold::*;
import ast::ident;
import ast::path_;
import ast::expr_path;

export add_new_extension;


//temporary, until 'position' shows up in the snapshot
fn position[T](&T x, &vec[T] v) -> option::t[uint] {
    let uint i = 0u;
    while (i < len(v)) {
        if (x == v.(i)) { ret some[uint](i); }
        i += 1u;
    }
    ret none[uint];
}

// substitute, in a position that's required to be an ident
fn subst_ident(&ext_ctxt cx, &vec[@ast::expr] args, 
               @vec[ident] param_names, &ident i, ast_fold fld) -> ident {
    alt (position(i, *param_names)) {
        case (some[uint](?idx)) {
            ret expr_to_ident(cx, args.(idx), 
                              "This argument is expanded as an "
                              + "identifier; it must be one.");
        }
        case (none[uint]) {
            ret i;
        }
    }
}

fn subst_path(&ext_ctxt cx, &vec[@ast::expr] args, 
              @vec[ident] param_names, &path_ p, ast_fold fld) -> path_ {
    // Don't substitute into qualified names.
    if (len(p.types) > 0u || len(p.idents) != 1u) { ret p; }
    alt (position(p.idents.(0), *param_names)) {
        case (some[uint](?idx)) {
            alt (args.(idx).node) {
                case (expr_path(?new_path)) {
                    ret new_path.node;
                }
                case (_) {
                    cx.span_fatal(args.(idx).span,
                                "This argument is expanded as a path; "
                                + "it must be one.");
                }
            }
        }
        case (none[uint]) { ret p; }
    }
}


fn subst_expr(&ext_ctxt cx, &vec[@ast::expr] args, @vec[ident] param_names, 
              &ast::expr_ e, ast_fold fld, 
              fn(&ast::expr_, ast_fold) -> ast::expr_ orig) -> ast::expr_ {
    ret alt(e) {
        case (expr_path(?p)){
            // Don't substitute into qualified names.
            if (len(p.node.types) > 0u || len(p.node.idents) != 1u) { e }
            alt (position(p.node.idents.(0), *param_names)) {
                case (some[uint](?idx)) {
                    args.(idx).node
                }
                case (none[uint]) { e }
            }
        }
        case (_) { orig(e,fld) }
    }
}


fn add_new_extension(&ext_ctxt cx, span sp, &vec[@ast::expr] args,
                     option::t[str] body) -> tup(str, syntax_extension) {
    if (len(args) < 2u) {
        cx.span_fatal(sp, "malformed extension description");
    }

    fn generic_extension(&ext_ctxt cx, span sp, &vec[@ast::expr] args,
                         option::t[str] body, @vec[ident] param_names,
                         @ast::expr dest_form) -> @ast::expr {
        if (len(args) != len(*param_names)) {
            cx.span_fatal(sp, #fmt("extension expects %u arguments, got %u",
                                 len(*param_names), len(args)));
        }

        auto afp = default_ast_fold();
        auto f_pre = 
            rec(fold_ident = bind subst_ident(cx, args, param_names, _, _),
                fold_path = bind subst_path(cx, args, param_names, _, _),
                fold_expr = bind subst_expr(cx, args, param_names, _, _,
                                            afp.fold_expr)
                with *afp);
        auto f = make_fold(f_pre);
        auto result = f.fold_expr(dest_form);
        dummy_out(f); //temporary: kill circular reference
        ret result;
        
    }

    let vec[ident] param_names = vec::empty[ident]();
    let uint idx = 1u;
    while(1u+idx < len(args)) {
        param_names +=
            [expr_to_ident(cx, args.(idx),
                           "this parameter name must be an identifier.")];
        idx += 1u;
    }

    ret tup(expr_to_str(cx, args.(0), "first arg must be a literal string."),
            normal(bind generic_extension(_,_,_,_,@param_names,
                                          args.(len(args)-1u))));
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
