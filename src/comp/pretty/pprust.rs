
import std::uint;
import std::vec;
import std::str;
import std::io;
import std::option;
import driver::session::session;
import front::lexer;
import front::codemap;
import front::codemap::codemap;
import front::ast;
import middle::ty;
import util::common;
import option::some;
import option::none;
import pp::printer;
import pp::break_offset;
import pp::word;
import pp::huge_word;
import pp::zero_word;
import pp::space;
import pp::zerobreak;
import pp::hardbreak;
import pp::breaks;
import pp::consistent;
import pp::inconsistent;
import pp::eof;
import ppaux::*;

fn print_crate(session sess, @ast::crate crate, str filename,
               io::writer out, mode mode) {
    let vec[pp::breaks] boxes = [];
    auto r = lexer::gather_comments_and_literals(sess, filename);
    auto s =
        @rec(s=pp::mk_printer(out, default_columns),
             cm=some(sess.get_codemap()),
             comments=some(r.cmnts),
             literals=some(r.lits),
             mutable cur_cmnt=0u,
             mutable cur_lit=0u,
             mutable boxes=boxes,
             mode=mode);
    print_inner_attributes(s, crate.node.attrs);
    print_mod(s, crate.node.module);
    eof(s.s);
}

fn ty_to_str(&ast::ty ty) -> str { be to_str(ty, print_type); }

fn pat_to_str(&@ast::pat pat) -> str { be to_str(pat, print_pat); }

fn expr_to_str(&@ast::expr e) -> str { be to_str(e, print_expr); }

fn stmt_to_str(&ast::stmt s) -> str { be to_str(s, print_stmt); }

fn item_to_str(&@ast::item i) -> str { be to_str(i, print_item); }

fn path_to_str(&ast::path p) -> str { be to_str(p, print_path); }

fn fun_to_str(&ast::_fn f, str name, vec[ast::ty_param] params) -> str {
    auto writer = io::string_writer();
    auto s = rust_printer(writer.get_writer());
    print_fn(s, f.decl, f.proto, name, params);
    eof(s.s);
    ret writer.get_str();
}

fn block_to_str(&ast::block blk) -> str {
    auto writer = io::string_writer();
    auto s = rust_printer(writer.get_writer());
    // containing cbox, will be closed by print-block at }

    cbox(s, indent_unit);
    // head-ibox, will be closed by print-block after {

    ibox(s, 0u);
    print_block(s, blk);
    eof(s.s);
    ret writer.get_str();
}

fn cbox(&ps s, uint u) {
    vec::push(s.boxes, pp::consistent);
    pp::cbox(s.s, u);
}

fn box(&ps s, uint u, pp::breaks b) {
    vec::push(s.boxes, b);
    pp::box(s.s, u, b);
}

fn word_nbsp(&ps s, str w) { word(s.s, w); word(s.s, " "); }

fn word_space(&ps s, str w) { word(s.s, w); space(s.s); }

fn popen(&ps s) { word(s.s, "("); }

fn pclose(&ps s) { word(s.s, ")"); }

fn head(&ps s, str w) {
    // outer-box is consistent
    cbox(s, indent_unit);
    // head-box is inconsistent
    ibox(s, str::char_len(w) + 1u);
    // keyword that starts the head
    word_nbsp(s, w);
}

fn bopen(&ps s) {
    word(s.s, "{");
    end(s); // close the head-box

}

fn bclose(&ps s, common::span span) {
    maybe_print_comment(s, span.hi);
    break_offset(s.s, 1u, -(indent_unit as int));
    word(s.s, "}");
    end(s); // close the outer-box

}

fn hardbreak_if_not_eof(&ps s) {
    if (s.s.last_token() != pp::EOF) {
        hardbreak(s.s);
    }
}

fn space_if_not_hardbreak(&ps s) {
    if (s.s.last_token() != pp::hardbreak_tok()) {
        space(s.s);
    }
}

// Synthesizes a comment that was not textually present in the original source
// file.
fn synth_comment(&ps s, str text) {
    word(s.s, "/*");
    space(s.s);
    word(s.s, text);
    space(s.s);
    word(s.s, "*/");
}

fn commasep[IN](&ps s, breaks b, vec[IN] elts, fn(&ps, &IN)  op) {
    box(s, 0u, b);
    auto first = true;
    for (IN elt in elts) {
        if (first) { first = false; } else { word_space(s, ","); }
        op(s, elt);
    }
    end(s);
}

fn commasep_cmnt[IN](&ps s, breaks b, vec[IN] elts, fn(&ps, &IN)  op,
                     fn(&IN) -> common::span  get_span) {
    box(s, 0u, b);
    auto len = vec::len[IN](elts);
    auto i = 0u;
    for (IN elt in elts) {
        maybe_print_comment(s, get_span(elt).hi);
        op(s, elt);
        i += 1u;
        if (i < len) {
            word(s.s, ",");
            maybe_print_trailing_comment(s, get_span(elt),
                                         some(get_span(elts.(i)).hi));
            space_if_not_hardbreak(s);
        }
    }
    end(s);
}

fn commasep_exprs(&ps s, breaks b, vec[@ast::expr] exprs) {
    fn expr_span(&@ast::expr expr) -> common::span { ret expr.span; }
    commasep_cmnt(s, b, exprs, print_expr, expr_span);
}

fn print_mod(&ps s, ast::_mod _mod) {
    for (@ast::view_item vitem in _mod.view_items) {
        print_view_item(s, vitem);
    }
    for (@ast::item item in _mod.items) {
        // Mod-level item printing we're a little more space-y about.

        hardbreak_if_not_eof(s);
        print_item(s, item);
    }
    print_remaining_comments(s);
}

fn print_boxed_type(&ps s, &@ast::ty ty) { print_type(s, *ty); }

fn print_type(&ps s, &ast::ty ty) {
    maybe_print_comment(s, ty.span.lo);
    ibox(s, 0u);
    alt (ty.node) {
        case (ast::ty_nil) { word(s.s, "()"); }
        case (ast::ty_bool) { word(s.s, "bool"); }
        case (ast::ty_bot) { word(s.s, "!"); }
        case (ast::ty_int) { word(s.s, "int"); }
        case (ast::ty_uint) { word(s.s, "uint"); }
        case (ast::ty_float) { word(s.s, "float"); }
        case (ast::ty_machine(?tm)) { word(s.s, common::ty_mach_to_str(tm)); }
        case (ast::ty_char) { word(s.s, "char"); }
        case (ast::ty_str) { word(s.s, "str"); }
        case (ast::ty_box(?mt)) { word(s.s, "@"); print_mt(s, mt); }
        case (ast::ty_vec(?mt)) {
            word(s.s, "vec[");
            print_mt(s, mt);
            word(s.s, "]");
        }
        case (ast::ty_ivec(?mt)) {
            print_type(s, *mt.ty);
            word(s.s, "[");
            print_mutability(s, mt.mut);
            word(s.s, "]");
        }
        case (ast::ty_port(?t)) {
            word(s.s, "port[");
            print_type(s, *t);
            word(s.s, "]");
        }
        case (ast::ty_chan(?t)) {
            word(s.s, "chan[");
            print_type(s, *t);
            word(s.s, "]");
        }
        case (ast::ty_type) { word(s.s, "type"); }
        case (ast::ty_tup(?elts)) {
            word(s.s, "tup");
            popen(s);
            commasep(s, inconsistent, elts, print_mt);
            pclose(s);
        }
        case (ast::ty_rec(?fields)) {
            word(s.s, "rec");
            popen(s);
            fn print_field(&ps s, &ast::ty_field f) {
                cbox(s, indent_unit);
                print_mt(s, f.node.mt);
                space(s.s);
                word(s.s, f.node.ident);
                end(s);
            }
            fn get_span(&ast::ty_field f) -> common::span { ret f.span; }
            commasep_cmnt(s, consistent, fields, print_field, get_span);
            pclose(s);
        }
        case (ast::ty_obj(?methods)) {
            head(s, "obj");
            bopen(s);
            for (ast::ty_method m in methods) {
                hardbreak(s.s);
                cbox(s, indent_unit);
                maybe_print_comment(s, m.span.lo);
                print_ty_fn(s, m.node.proto, some(m.node.ident),
                            m.node.inputs, m.node.output, m.node.cf,
                            m.node.constrs);
                word(s.s, ";");
                end(s);
            }
            bclose(s, ty.span);
        }
        case (ast::ty_fn(?proto, ?inputs, ?output, ?cf, ?constrs)) {
            print_ty_fn(s, proto, none[str], inputs, output, cf, constrs);
        }
        case (ast::ty_path(?path, _)) { print_path(s, path); }
    }
    end(s);
}

fn print_item(&ps s, &@ast::item item) {
    hardbreak_if_not_eof(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    alt (item.node) {
        case (ast::item_const(?ty, ?expr)) {
            head(s, "const");
            print_type(s, *ty);
            space(s.s);
            word_space(s, item.ident);
            end(s); // end the head-ibox

            word_space(s, "=");
            print_expr(s, expr);
            word(s.s, ";");
            end(s); // end the outer cbox

        }
        case (ast::item_fn(?_fn, ?typarams)) {
            print_fn(s, _fn.decl, _fn.proto, item.ident, typarams);
            word(s.s, " ");
            print_block(s, _fn.body);
        }
        case (ast::item_mod(?_mod)) {
            head(s, "mod");
            word_nbsp(s, item.ident);
            bopen(s);
            print_inner_attributes(s, item.attrs);
            for (@ast::item itm in _mod.items) { print_item(s, itm); }
            bclose(s, item.span);
        }
        case (ast::item_native_mod(?nmod)) {
            head(s, "native");
            alt (nmod.abi) {
                case (ast::native_abi_rust) { word_nbsp(s, "\"rust\""); }
                case (ast::native_abi_cdecl) { word_nbsp(s, "\"cdecl\""); }
                case (ast::native_abi_rust_intrinsic) {
                    word_nbsp(s, "\"rust-intrinsic\"");
                }
            }
            word_nbsp(s, "mod");
            word_nbsp(s, item.ident);
            bopen(s);
            for (@ast::native_item item in nmod.items) {
                hardbreak(s.s);
                ibox(s, indent_unit);
                maybe_print_comment(s, item.span.lo);
                alt (item.node) {
                    case (ast::native_item_ty(?id, _)) {
                        word_nbsp(s, "type");
                        word(s.s, id);
                    }
                    case (ast::native_item_fn(?id, ?lname, ?decl, ?typarams,
                                              _, _)) {
                        print_fn(s, decl, ast::proto_fn, id, typarams);
                        alt (lname) {
                            case (none) { }
                            case (some(?ss)) {
                                space(s.s);
                                word_space(s, "=");
                                print_string(s, ss);
                            }
                        }
                        end(s); // end head-ibox

                        end(s); // end the outer fn box

                    }
                }
                word(s.s, ";");
                end(s);
            }
            bclose(s, item.span);
        }
        case (ast::item_ty(?ty, ?params)) {
            ibox(s, indent_unit);
            ibox(s, 0u);
            word_nbsp(s, "type");
            word(s.s, item.ident);
            print_type_params(s, params);
            end(s); // end the inner ibox

            space(s.s);
            word_space(s, "=");
            print_type(s, *ty);
            word(s.s, ";");
            end(s); // end the outer ibox

            break_offset(s.s, 0u, 0);
        }
        case (ast::item_tag(?variants, ?params)) {
            head(s, "tag");
            word(s.s, item.ident);
            print_type_params(s, params);
            space(s.s);
            bopen(s);
            for (ast::variant v in variants) {
                space(s.s);
                maybe_print_comment(s, v.span.lo);
                word(s.s, v.node.name);
                if (vec::len(v.node.args) > 0u) {
                    popen(s);
                    fn print_variant_arg(&ps s, &ast::variant_arg arg) {
                        print_type(s, *arg.ty);
                    }
                    commasep(s, consistent, v.node.args, print_variant_arg);
                    pclose(s);
                }
                word(s.s, ";");
                maybe_print_trailing_comment(s, v.span, none[uint]);
            }
            bclose(s, item.span);
        }
        case (ast::item_obj(?_obj, ?params, _)) {
            head(s, "obj");
            word(s.s, item.ident);
            print_type_params(s, params);
            popen(s);
            fn print_field(&ps s, &ast::obj_field field) {
                ibox(s, indent_unit);
                print_mutability(s, field.mut);
                print_type(s, *field.ty);
                space(s.s);
                word(s.s, field.ident);
                end(s);
            }
            fn get_span(&ast::obj_field f) -> common::span { ret f.ty.span; }
            commasep_cmnt(s, consistent, _obj.fields, print_field, get_span);
            pclose(s);
            space(s.s);
            bopen(s);
            for (@ast::method meth in _obj.methods) {
                let vec[ast::ty_param] typarams = [];
                hardbreak(s.s);
                maybe_print_comment(s, meth.span.lo);
                print_fn(s, meth.node.meth.decl, meth.node.meth.proto,
                         meth.node.ident, typarams);
                word(s.s, " ");
                print_block(s, meth.node.meth.body);
            }
            alt (_obj.dtor) {
                case (some(?dtor)) {
                    head(s, "drop");
                    print_block(s, dtor.node.meth.body);
                }
                case (_) { }
            }
            bclose(s, item.span);
        }
    }

    // Print the node ID if necessary. TODO: type as well.
    alt (s.mode) {
        case (mo_identified) {
            space(s.s);
            synth_comment(s, uint::to_str(item.ann.id, 10u));
        }
        case (_) {/* no-op */ }
    }
}

fn print_outer_attributes(&ps s, vec[ast::attribute] attrs) {
    auto count = 0;
    for (ast::attribute attr in attrs) {
        alt (attr.node.style) {
            case (ast::attr_outer) { print_attribute(s, attr); count += 1; }
            case (_) {/* fallthrough */ }
        }
    }
    if (count > 0) { hardbreak(s.s); }
}

fn print_inner_attributes(&ps s, vec[ast::attribute] attrs) {
    auto count = 0;
    for (ast::attribute attr in attrs) {
        alt (attr.node.style) {
            case (ast::attr_inner) {
                print_attribute(s, attr);
                word(s.s, ";");
                count += 1;
            }
            case (_) { /* fallthrough */ }
        }
    }
    if (count > 0) { hardbreak(s.s); }
}

fn print_attribute(&ps s, &ast::attribute attr) {
    hardbreak(s.s);
    maybe_print_comment(s, attr.span.lo);
    word(s.s, "#[");
    print_meta_item(s, @attr.node.value);
    word(s.s, "]");
}

fn print_stmt(&ps s, &ast::stmt st) {
    maybe_print_comment(s, st.span.lo);
    alt (st.node) {
        case (ast::stmt_decl(?decl, _)) { print_decl(s, decl); }
        case (ast::stmt_expr(?expr, _)) {
            space_if_not_hardbreak(s);
            print_expr(s, expr);
        }
    }
    if (front::parser::stmt_ends_with_semi(st)) { word(s.s, ";"); }
    maybe_print_trailing_comment(s, st.span, none[uint]);
}

fn print_block(&ps s, ast::block blk) {
    maybe_print_comment(s, blk.span.lo);
    bopen(s);
    for (@ast::stmt st in blk.node.stmts) { print_stmt(s, *st) }
    alt (blk.node.expr) {
        case (some(?expr)) {
            space_if_not_hardbreak(s);
            print_expr(s, expr);
            maybe_print_trailing_comment(s, expr.span, some(blk.span.hi));
        }
        case (_) { }
    }
    bclose(s, blk.span);

    // Print the node ID if necessary: TODO: type as well.
    alt (s.mode) {
        case (mo_identified) {
            space(s.s);
            synth_comment(s, "block " + uint::to_str(blk.node.a.id, 10u));
        }
        case (_) {/* no-op */ }
    }
}

fn print_if(&ps s, &@ast::expr test, &ast::block block,
            &option::t[@ast::expr] elseopt, bool chk) {
    head(s, "if");
    if (chk) {
        word_nbsp(s, "check");
    }
    popen(s);
    print_expr(s, test);
    pclose(s);
    space(s.s);
    print_block(s, block);
    fn do_else(&ps s, option::t[@ast::expr] els) {
        alt (els) {
            case (some(?_else)) {
                alt (_else.node) {
                    case (
                          // "another else-if"
                          ast::expr_if(?i, ?t, ?e, _)) {
                        cbox(s, indent_unit - 1u);
                        ibox(s, 0u);
                        word(s.s, " else if ");
                        popen(s);
                        print_expr(s, i);
                        pclose(s);
                        space(s.s);
                        print_block(s, t);
                        do_else(s, e);
                    }
                    case (
                          // "final else"
                          ast::expr_block(?b, _)) {
                        cbox(s, indent_unit - 1u);
                        ibox(s, 0u);
                        word(s.s, " else ");
                        print_block(s, b);
                    }
                }
            }
            case (_) {/* fall through */ }
        }
    }
    do_else(s, elseopt);
}

fn print_expr(&ps s, &@ast::expr expr) {
    maybe_print_comment(s, expr.span.lo);
    ibox(s, indent_unit);
    alt (s.mode) {
        case (mo_untyped) {/* no-op */ }
        case (mo_typed(_)) { popen(s); }
        case (mo_identified) { popen(s); }
    }
    alt (expr.node) {
        case (ast::expr_vec(?exprs, ?mut, ?kind, _)) {
            ibox(s, indent_unit);
            alt (kind) {
                case (ast::sk_rc) { word(s.s, "["); }
                case (ast::sk_unique) { word(s.s, "~["); }
            }
            if (mut == ast::mut) { word_nbsp(s, "mutable"); }
            commasep_exprs(s, inconsistent, exprs);
            word(s.s, "]");
            end(s);
        }
        case (ast::expr_tup(?exprs, _)) {
            fn printElt(&ps s, &ast::elt elt) {
                ibox(s, indent_unit);
                if (elt.mut == ast::mut) { word_nbsp(s, "mutable"); }
                print_expr(s, elt.expr);
                end(s);
            }
            fn get_span(&ast::elt elt) -> common::span { ret elt.expr.span; }
            word(s.s, "tup");
            popen(s);
            commasep_cmnt(s, inconsistent, exprs, printElt, get_span);
            pclose(s);
        }
        case (ast::expr_rec(?fields, ?wth, _)) {
            fn print_field(&ps s, &ast::field field) {
                ibox(s, indent_unit);
                if (field.node.mut == ast::mut) { word_nbsp(s, "mutable"); }
                word(s.s, field.node.ident);
                word(s.s, "=");
                print_expr(s, field.node.expr);
                end(s);
            }
            fn get_span(&ast::field field) -> common::span { ret field.span; }
            word(s.s, "rec");
            popen(s);
            commasep_cmnt(s, consistent, fields, print_field, get_span);
            alt (wth) {
                case (some(?expr)) {
                    if (vec::len(fields) > 0u) { space(s.s); }
                    ibox(s, indent_unit);
                    word_space(s, "with");
                    print_expr(s, expr);
                    end(s);
                }
                case (_) { }
            }
            pclose(s);
        }
        case (ast::expr_call(?func, ?args, _)) {
            print_expr(s, func);
            popen(s);
            commasep_exprs(s, inconsistent, args);
            pclose(s);
        }
        case (ast::expr_self_method(?ident, _)) {
            word(s.s, "self.");
            print_ident(s, ident);
        }
        case (ast::expr_bind(?func, ?args, _)) {
            fn print_opt(&ps s, &option::t[@ast::expr] expr) {
                alt (expr) {
                    case (some(?expr)) { print_expr(s, expr); }
                    case (_) { word(s.s, "_"); }
                }
            }
            word_nbsp(s, "bind");
            print_expr(s, func);
            popen(s);
            commasep(s, inconsistent, args, print_opt);
            pclose(s);
        }
        case (ast::expr_spawn(_, _, ?e, ?es, _)) {
            word_nbsp(s, "spawn");
            print_expr(s, e);
            popen(s);
            commasep_exprs(s, inconsistent, es);
            pclose(s);
        }
        case (ast::expr_binary(?op, ?lhs, ?rhs, _)) {
            auto prec = operator_prec(op);
            print_maybe_parens(s, lhs, prec);
            space(s.s);
            word_space(s, ast::binop_to_str(op));
            print_maybe_parens(s, rhs, prec + 1);
        }
        case (ast::expr_unary(?op, ?expr, _)) {
            word(s.s, ast::unop_to_str(op));
            print_maybe_parens(s, expr, front::parser::unop_prec);
        }
        case (ast::expr_lit(?lit, _)) { print_literal(s, lit); }
        case (ast::expr_cast(?expr, ?ty, _)) {
            print_maybe_parens(s, expr, front::parser::as_prec);
            space(s.s);
            word_space(s, "as");
            print_type(s, *ty);
        }
        case (ast::expr_if(?test, ?block, ?elseopt, _)) {
            print_if(s, test, block, elseopt, false);
        }
        case (ast::expr_if_check(?test, ?block, ?elseopt, _)) {
            print_if(s, test, block, elseopt, true);
        }
        case (ast::expr_while(?test, ?block, _)) {
            head(s, "while");
            popen(s);
            print_expr(s, test);
            pclose(s);
            space(s.s);
            print_block(s, block);
        }
        case (ast::expr_for(?decl, ?expr, ?block, _)) {
            head(s, "for");
            popen(s);
            print_for_decl(s, decl);
            space(s.s);
            word_space(s, "in");
            print_expr(s, expr);
            pclose(s);
            space(s.s);
            print_block(s, block);
        }
        case (ast::expr_for_each(?decl, ?expr, ?block, _)) {
            head(s, "for each");
            popen(s);
            print_for_decl(s, decl);
            space(s.s);
            word_space(s, "in");
            print_expr(s, expr);
            pclose(s);
            space(s.s);
            print_block(s, block);
        }
        case (ast::expr_do_while(?block, ?expr, _)) {
            head(s, "do");
            space(s.s);
            print_block(s, block);
            space(s.s);
            word_space(s, "while");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_alt(?expr, ?arms, _)) {
            head(s, "alt");
            popen(s);
            print_expr(s, expr);
            pclose(s);
            space(s.s);
            bopen(s);
            for (ast::arm arm in arms) {
                space(s.s);
                head(s, "case");
                popen(s);
                print_pat(s, arm.pat);
                pclose(s);
                space(s.s);
                print_block(s, arm.block);
            }
            bclose(s, expr.span);
        }
        case (ast::expr_fn(?f, _)) {
            head(s, "fn");
            print_fn_args_and_ret(s, f.decl);
            space(s.s);
            print_block(s, f.body);
        }
        case (ast::expr_block(?block, _)) {
            // containing cbox, will be closed by print-block at }

            cbox(s, indent_unit);
            // head-box, will be closed by print-block after {

            ibox(s, 0u);
            print_block(s, block);
        }
        case (ast::expr_move(?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "<-");
            print_expr(s, rhs);
        }
        case (ast::expr_assign(?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "=");
            print_expr(s, rhs);
        }
        case (ast::expr_swap(?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "<->");
            print_expr(s, rhs);
        }
        case (ast::expr_assign_op(?op, ?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word(s.s, ast::binop_to_str(op));
            word_space(s, "=");
            print_expr(s, rhs);
        }
        case (ast::expr_send(?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "<|");
            print_expr(s, rhs);
        }
        case (ast::expr_recv(?lhs, ?rhs, _)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "|>");
            print_expr(s, rhs);
        }
        case (ast::expr_field(?expr, ?id, _)) {
            print_expr(s, expr);
            word(s.s, ".");
            word(s.s, id);
        }
        case (ast::expr_index(?expr, ?index, _)) {
            print_expr(s, expr);
            word(s.s, ".");
            popen(s);
            print_expr(s, index);
            pclose(s);
        }
        case (ast::expr_path(?path, _)) { print_path(s, path); }
        case (ast::expr_fail(_, ?str)) {
            word(s.s, "fail");
            alt (str) {
                case (some(?msg)) { word(s.s, #fmt("\"%s\"", msg)); }
                case (_) { }
            }
        }
        case (ast::expr_break(_)) { word(s.s, "break"); }
        case (ast::expr_cont(_)) { word(s.s, "cont"); }
        case (ast::expr_ret(?result, _)) {
            word(s.s, "ret");
            alt (result) {
                case (some(?expr)) { word(s.s, " "); print_expr(s, expr); }
                case (_) { }
            }
        }
        case (ast::expr_put(?result, _)) {
            word(s.s, "put");
            alt (result) {
                case (some(?expr)) { word(s.s, " "); print_expr(s, expr); }
                case (_) { }
            }
        }
        case (ast::expr_be(?result, _)) {
            word_nbsp(s, "be");
            print_expr(s, result);
        }
        case (ast::expr_log(?lvl, ?expr, _)) {
            alt (lvl) {
                case (1) { word_nbsp(s, "log"); }
                case (0) { word_nbsp(s, "log_err"); }
            }
            print_expr(s, expr);
        }
        case (ast::expr_check(?expr, _)) {
            word_nbsp(s, "check");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_assert(?expr, _)) {
            word_nbsp(s, "assert");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_ext(?path, ?args, ?body, _, _)) {
            word(s.s, "#");
            print_path(s, path);
            if (vec::len(args) > 0u) {
                popen(s);
                commasep_exprs(s, inconsistent, args);
                pclose(s);
            }
            // FIXME: extension 'body'

        }
        case (ast::expr_port(_)) { word(s.s, "port"); popen(s); pclose(s); }
        case (ast::expr_chan(?expr, _)) {
            word(s.s, "chan");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_anon_obj(_, _, _, _)) {
            word(s.s, "anon obj");
            // FIXME (issue #499): nicer pretty-printing of anon objs

        }
    }
    // Print the type or node ID if necessary.

    alt (s.mode) {
        case (mo_untyped) {/* no-op */ }
        case (mo_typed(?tcx)) {
            space(s.s);
            word(s.s, "as");
            space(s.s);
            word(s.s, ppaux::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
            pclose(s);
        }
        case (mo_identified) {
            space(s.s);
            synth_comment(s, uint::to_str(ty::expr_ann(expr).id, 10u));
            pclose(s);
        }
    }
    end(s);
}

fn print_decl(&ps s, &@ast::decl decl) {
    maybe_print_comment(s, decl.span.lo);
    alt (decl.node) {
        case (ast::decl_local(?loc)) {
            space_if_not_hardbreak(s);
            ibox(s, indent_unit);
            alt (loc.node.ty) {
                case (some(?ty)) {
                    word_nbsp(s, "let");
                    print_type(s, *ty);
                    space(s.s);
                }
                case (_) {
                    word_nbsp(s, "auto");

                    // Print the type or node ID if necessary.
                    alt (s.mode) {
                        case (mo_untyped) {/* no-op */ }
                        case (mo_typed(?tcx)) {
                            auto lty = ty::ann_to_type(tcx, loc.node.ann);
                            word_space(s, ppaux::ty_to_str(tcx, lty));
                        }
                        case (mo_identified) {/* no-op */ }
                    }
                }
            }
            word(s.s, loc.node.ident);
            alt (loc.node.init) {
                case (some(?init)) {
                    space(s.s);
                    alt (init.op) {
                        case (ast::init_assign) { word_space(s, "="); }
                        case (ast::init_move) { word_space(s, "<-"); }
                        case (ast::init_recv) { word_space(s, "|>"); }
                    }
                    print_expr(s, init.expr);
                }
                case (_) { }
            }
            end(s);
        }
        case (ast::decl_item(?item)) { print_item(s, item); }
    }
}

fn print_ident(&ps s, &ast::ident ident) { word(s.s, ident); }

fn print_for_decl(&ps s, @ast::local loc) {
    print_type(s, *option::get(loc.node.ty));
    space(s.s);
    word(s.s, loc.node.ident);
}

fn print_path(&ps s, &ast::path path) {
    maybe_print_comment(s, path.span.lo);
    auto first = true;
    for (str id in path.node.idents) {
        if (first) { first = false; } else { word(s.s, "::"); }
        word(s.s, id);
    }
    if (vec::len(path.node.types) > 0u) {
        word(s.s, "[");
        commasep(s, inconsistent, path.node.types, print_boxed_type);
        word(s.s, "]");
    }
}

fn print_pat(&ps s, &@ast::pat pat) {
    maybe_print_comment(s, pat.span.lo);
    alt (pat.node) {
        case (ast::pat_wild(_)) { word(s.s, "_"); }
        case (ast::pat_bind(?id, _, _)) { word(s.s, "?" + id); }
        case (ast::pat_lit(?lit, _)) { print_literal(s, lit); }
        case (ast::pat_tag(?path, ?args, _)) {
            print_path(s, path);
            if (vec::len(args) > 0u) {
                popen(s);
                commasep(s, inconsistent, args, print_pat);
                pclose(s);
            }
        }
    }

    // Print the node ID if necessary. TODO: type as well.
    alt (s.mode) {
        case (mo_identified) {
            space(s.s);
            synth_comment(s, uint::to_str(ty::pat_ann(pat).id, 10u));
        }
        case (_) {/* no-op */ }
    }
}

fn print_fn(&ps s, ast::fn_decl decl, ast::proto proto, str name,
            vec[ast::ty_param] typarams) {
    alt (decl.purity) {
        case (ast::impure_fn) {
            if (proto == ast::proto_iter) {
                head(s, "iter");
            } else { head(s, "fn"); }
        }
        case (_) { head(s, "pred"); }
    }
    word(s.s, name);
    print_type_params(s, typarams);
    print_fn_args_and_ret(s, decl);
}

fn print_fn_args_and_ret(&ps s, &ast::fn_decl decl) {
    popen(s);
    fn print_arg(&ps s, &ast::arg x) {
        ibox(s, indent_unit);
        print_alias(s, x.mode);
        print_type(s, *x.ty);
        space(s.s);
        word(s.s, x.ident);
        end(s);
    }
    commasep(s, inconsistent, decl.inputs, print_arg);
    pclose(s);
    maybe_print_comment(s, decl.output.span.lo);
    if (decl.output.node != ast::ty_nil) {
        space_if_not_hardbreak(s);
        word_space(s, "->");
        print_type(s, *decl.output);
    }
}

fn print_alias(&ps s, ast::mode m) {
    alt (m) {
        case (ast::alias(true)) { word_space(s, "&mutable"); }
        case (ast::alias(false)) { word(s.s, "&"); }
        case (ast::val) { }
    }
}

fn print_type_params(&ps s, &vec[ast::ty_param] params) {
    if (vec::len(params) > 0u) {
        word(s.s, "[");
        fn printParam(&ps s, &ast::ty_param param) { word(s.s, param); }
        commasep(s, inconsistent, params, printParam);
        word(s.s, "]");
    }
}

fn print_meta_item(&ps s, &@ast::meta_item item) {
    ibox(s, indent_unit);
    word_space(s, item.node.key);
    word_space(s, "=");
    print_string(s, item.node.value);
    end(s);
}

fn print_view_item(&ps s, &@ast::view_item item) {
    hardbreak_if_not_eof(s);
    maybe_print_comment(s, item.span.lo);
    alt (item.node) {
        case (ast::view_item_use(?id, ?mta, _, _)) {
            head(s, "use");
            word(s.s, id);
            if (vec::len(mta) > 0u) {
                popen(s);
                commasep(s, consistent, mta, print_meta_item);
                pclose(s);
            }
        }
        case (ast::view_item_import(?id, ?ids, _)) {
            head(s, "import");
            if (!str::eq(id, ids.(vec::len(ids) - 1u))) {
                word_space(s, id);
                word_space(s, "=");
            }
            auto first = true;
            for (str elt in ids) {
                if (first) { first = false; } else { word(s.s, "::"); }
                word(s.s, elt);
            }
        }
        case (ast::view_item_import_glob(?ids, _)) {
            head(s, "import");
            auto first = true;
            for (str elt in ids) {
                if (first) { first = false; } else { word(s.s, "::"); }
                word(s.s, elt);
            }
            word(s.s, "::*");
        }
        case (ast::view_item_export(?id)) {
            head(s, "export");
            word(s.s, id);
        }
    }
    word(s.s, ";");
    end(s); // end inner head-block

    end(s); // end outer head-block

}


// FIXME: The fact that this builds up the table anew for every call is
// not good. Eventually, table should be a const.
fn operator_prec(ast::binop op) -> int {
    for (front::parser::op_spec spec in front::parser::prec_table()) {
        if (spec.op == op) { ret spec.prec; }
    }
    fail;
}

fn print_maybe_parens(&ps s, &@ast::expr expr, int outer_prec) {
    auto add_them;
    alt (expr.node) {
        case (ast::expr_binary(?op, _, _, _)) {
            add_them = operator_prec(op) < outer_prec;
        }
        case (ast::expr_cast(_, _, _)) {
            add_them = front::parser::as_prec < outer_prec;
        }
        case (_) { add_them = false; }
    }
    if (add_them) { popen(s); }
    print_expr(s, expr);
    if (add_them) { pclose(s); }
}

fn print_mutability(&ps s, &ast::mutability mut) {
    alt (mut) {
        case (ast::mut) { word_nbsp(s, "mutable"); }
        case (ast::maybe_mut) { word_nbsp(s, "mutable?"); }
        case (ast::imm) {/* nothing */ }
    }
}

fn print_mt(&ps s, &ast::mt mt) {
    print_mutability(s, mt.mut);
    print_type(s, *mt.ty);
}

fn print_ty_fn(&ps s, &ast::proto proto, &option::t[str] id,
               &vec[ast::ty_arg] inputs, &@ast::ty output,
               &ast::controlflow cf, &vec[@ast::constr] constrs) {
    ibox(s, indent_unit);
    if (proto == ast::proto_fn) {
        word(s.s, "fn");
    } else { word(s.s, "iter"); }
    alt (id) {
        case (some(?id)) { word(s.s, " "); word(s.s, id); }
        case (_) { }
    }
    zerobreak(s.s);
    popen(s);
    fn print_arg(&ps s, &ast::ty_arg input) {
        print_alias(s, input.node.mode);
        print_type(s, *input.node.ty);
    }
    commasep(s, inconsistent, inputs, print_arg);
    pclose(s);
    maybe_print_comment(s, output.span.lo);
    if (output.node != ast::ty_nil) {
        space_if_not_hardbreak(s);
        ibox(s, indent_unit);
        word_space(s, "->");
        alt (cf) {
            case (ast::return) { print_type(s, *output); }
            case (ast::noreturn) { word_nbsp(s, "!"); }
        }
        end(s);
    }
    word_space(s, ast_constrs_str(constrs));
    end(s);
}

fn maybe_print_trailing_comment(&ps s, common::span span,
                                option::t[uint] next_pos) {
    auto cm;
    alt (s.cm) { case (some(?ccm)) { cm = ccm; } case (_) { ret; } }
    alt (next_comment(s)) {
        case (some(?cmnt)) {
            if (cmnt.style != lexer::trailing) { ret; }
            auto span_line = codemap::lookup_pos(cm, span.hi);
            auto comment_line = codemap::lookup_pos(cm, cmnt.pos);
            auto next = cmnt.pos + 1u;
            alt (next_pos) { case (none) { } case (some(?p)) { next = p; } }
            if (span.hi < cmnt.pos && cmnt.pos < next &&
                    span_line.line == comment_line.line) {
                print_comment(s, cmnt);
                s.cur_cmnt += 1u;
            }
        }
        case (_) { }
    }
}

fn print_remaining_comments(&ps s) {
    while (true) {
        alt (next_comment(s)) {
            case (some(?cmnt)) { print_comment(s, cmnt); s.cur_cmnt += 1u; }
            case (_) { break; }
        }
    }
}

fn in_cbox(&ps s) -> bool {
    auto len = vec::len(s.boxes);
    if (len == 0u) { ret false; }
    ret s.boxes.(len - 1u) == pp::consistent;
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
