use std;

import codemap::span;
import std::ivec;
import std::vec;
import std::option;
import vec::map;
import vec::len;
import option::some;
import option::none;

import base::syntax_extension;
import base::ext_ctxt;
import base::normal;
import base::expr_to_str;
import base::expr_to_ident;

import fold::*;
import ast::respan;
import ast::ident;
import ast::path;
import ast::path_;
import ast::expr_path;
import ast::expr_vec;
import ast::expr_mac;
import ast::mac_invoc;

export add_new_extension;

fn lookup(&(invk_binding)[] ibs, ident i) -> option::t[invk_binding] {
    for (invk_binding ib in ibs) {
        alt (ib) {
            case (ident_binding(?p_id, _)) { if (i == p_id) { ret some(ib); }}
            case (path_binding(?p_id, _)) { if (i == p_id) { ret some(ib); }}
            case (expr_binding(?p_id, _)) { if (i == p_id) { ret some(ib); }}
        }
    }
    ret none;
}

// substitute, in a position that's required to be an ident
fn subst_ident(&ext_ctxt cx, &(invk_binding)[] ibs, &ident i, ast_fold fld)
    -> ident {
    ret alt (lookup(ibs, i)) {
        case (some(ident_binding(_, ?a_id))) { a_id.node }
        case (some(path_binding(_, ?pth))) {
            cx.span_fatal(pth.span, "This argument is expanded as an "
                          + "identifier; it must be one.")
        }
        case (some(expr_binding(_, ?expr))) {
            cx.span_fatal(expr.span, "This argument is expanded as an "
                          + "identifier; it must be one.")
        }
        case (none) { i }
    }
}


fn subst_path(&ext_ctxt cx, &(invk_binding)[] ibs, &path_ p, ast_fold fld)
    -> path_ {
    // Don't substitute into qualified names.
    if (ivec::len(p.types) > 0u || ivec::len(p.idents) != 1u) { ret p; }
    ret alt (lookup(ibs, p.idents.(0))) {
        case (some(ident_binding(_, ?id))) { 
            rec(global=false, idents=~[id.node], types=~[]) 
        }
        case (some(path_binding(_, ?a_pth))) { a_pth.node }
        case (some(expr_binding(_, ?expr))) {
            cx.span_fatal(expr.span, "This argument is expanded as an "
                          + "path; it must be one.")
        }
        case (none) { p }
    }
}


fn subst_expr(&ext_ctxt cx, &(invk_binding)[] ibs, &ast::expr_ e, 
              ast_fold fld, fn(&ast::expr_, ast_fold) -> ast::expr_ orig) 
    -> ast::expr_ {
    ret alt(e) {
        case (expr_path(?p)){
            // Don't substitute into qualified names.
            if (ivec::len(p.node.types) > 0u || 
                ivec::len(p.node.idents) != 1u) { e }
            alt (lookup(ibs, p.node.idents.(0))) {
                case (some(ident_binding(_, ?id))) { 
                    expr_path(respan(id.span, 
                                     rec(global=false, 
                                         idents=~[id.node],types=~[])))
                }
                case (some(path_binding(_, ?a_pth))) { expr_path(*a_pth) }
                case (some(expr_binding(_, ?a_exp))) { a_exp.node }
                case (none) { orig(e,fld) }
            }
        }
        case (_) { orig(e,fld) }
    }
}

type pat_ext = rec((@ast::expr)[] invk, @ast::expr body);

// maybe box?
tag invk_binding {
    expr_binding(ident, @ast::expr);
    path_binding(ident, @ast::path);
    ident_binding(ident, ast::spanned[ident]);
}

fn path_to_ident(&path pth) -> option::t[ident] {
    if (ivec::len(pth.node.idents) == 1u 
        && ivec::len(pth.node.types) == 0u) {
        ret some(pth.node.idents.(0u));
    }
    ret none;
}

fn process_clause(&ext_ctxt cx, &mutable vec[pat_ext] pes,
                  &mutable option::t[str] macro_name, &path pth, 
                  &(@ast::expr)[] invoc_args, @ast::expr body) {
    let str clause_name = alt(path_to_ident(pth)) {
        case (some(?id)) { id }
        case (none) {
            cx.span_fatal(pth.span, "macro name must not be a path")
        }
    };
    if (macro_name == none) {
        macro_name = some(clause_name);
    } else if (macro_name != some(clause_name)) {
        cx.span_fatal(pth.span, "#macro can only introduce one name");
    }
    pes += [rec(invk=invoc_args, body=body)];
}


fn add_new_extension(&ext_ctxt cx, span sp, &(@ast::expr)[] args,
                     option::t[str] body) -> tup(str, syntax_extension) {
    let option::t[str] macro_name = none;
    let vec[pat_ext] pat_exts = [];
    for (@ast::expr arg in args) {
        alt(arg.node) {
            case(expr_vec(?elts, ?mut, ?seq_kind)) {
                
                if (ivec::len(elts) != 2u) {
                    cx.span_fatal((*arg).span, 
                                  "extension clause must consist of [" + 
                                  "macro invocation, expansion body]");
                }
                alt(elts.(0u).node) {
                    case(expr_mac(?mac)) {
                        alt (mac.node) {
                            case (mac_invoc(?pth, ?invoc_args, ?body)) {
                                process_clause(cx, pat_exts, macro_name,
                                               pth, invoc_args, elts.(1u));
                            }
                        }
                    }
                    case(_) {
                        cx.span_fatal(elts.(0u).span, "extension clause must"
                                      + " start with a macro invocation.");
                    }
                }
            }
            case(_) {
                    cx.span_fatal((*arg).span, "extension must be [clause, "
                                  + " ...]");
            }
        }
    }

    auto ext = bind generic_extension(_,_,_,_,@pat_exts);
    
    ret tup(alt (macro_name) {
                case (some(?id)) { id }
                case (none) { 
                    cx.span_fatal(sp, "macro definition must have "
                                  + "at least one clause")
                }
            },
            normal(ext));


    fn generic_extension(&ext_ctxt cx, span sp, &(@ast::expr)[] args,
                         option::t[str] body, @vec[pat_ext] clauses)
        -> @ast::expr {

        /* returns a list of bindings, or none if the match fails. */
        fn match_invk(@ast::expr pattern, @ast::expr argument)
            -> option::t[(invk_binding)[]] {
            auto pat = pattern.node;
            auto arg = argument.node;
            ret alt (pat) {
                case (expr_vec(?p_elts, _, _)) {
                    alt (arg) {
                        case (expr_vec(?a_elts, _, _)) {
                            if (ivec::len(p_elts) != ivec::len(a_elts)) { 
                                none[vec[invk_binding]]
                            }
                            let uint i = 0u;
                            let (invk_binding)[] res = ~[];
                            while (i < ivec::len(p_elts)) {
                                alt (match_invk(p_elts.(i), a_elts.(i))) {
                                    case (some(?v)) { res += v; }
                                    case (none) { ret none; }
                                }
                                i += 1u;
                            }
                            some(res)
                        }
                        case (_) { none }
                    }
                }
                case (expr_path(?p_pth)) {
                    alt (path_to_ident(p_pth)) {
                        case (some(?p_id)) {
                            /* let's bind! */
                            alt (arg) {
                                case (expr_path(?a_pth)) {
                                    alt (path_to_ident(a_pth)) {
                                        case (some(?a_id)) {
                                            some(~[ident_binding
                                                   (p_id, 
                                                    respan(argument.span,
                                                                 a_id))])
                                        }
                                        case (none) {
                                            some(~[path_binding(p_id, 
                                                                @a_pth)])
                                        }
                                    }
                                }
                                case (_) {
                                    some(~[expr_binding(p_id, argument)])
                                }
                            }
                        }
                        // FIXME this still compares on internal spans
                        case (_) { if(pat == arg) { some(~[]) } else { none }}
                    }
                }
                // FIXME this still compares on internal spans
                case (_) { if (pat == arg) { some(~[]) } else { none }}
            }
        }

        for (pat_ext pe in *clauses) {
            if (ivec::len(args) != ivec::len(pe.invk)) { cont; }
            let uint i = 0u;
            let (invk_binding)[] bindings = ~[];
            while (i < ivec::len(args)) {
                alt (match_invk(pe.invk.(i), args.(i))) {
                    case (some(?v)) { bindings += v; }
                    case (none) { cont }
                }
                i += 1u;
            }
            auto afp = default_ast_fold();
            auto f_pre =
                rec(fold_ident = bind subst_ident(cx, bindings, _, _),
                    fold_path = bind subst_path(cx, bindings, _, _),
                    fold_expr = bind subst_expr(cx, bindings, _, _,
                                                afp.fold_expr)
                with *afp);
            auto f = make_fold(f_pre);
            auto result = f.fold_expr(pe.body);
            dummy_out(f); //temporary: kill circular reference
            ret result;
        }
        cx.span_fatal(sp, "no clauses match macro invocation");
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
