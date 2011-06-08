import std::uint;
import std::vec;
import std::str;
import std::io;
import std::option;
import driver::session::session;
import front::ast;
import front::lexer;
import front::codemap;
import front::codemap::codemap;
import middle::ty;
import util::common;
import pp;

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

const uint indent_unit = 4u;
const uint default_columns = 78u;

tag mode {
    mo_untyped;
    mo_typed(ty::ctxt);
    mo_identified;
}

type ps = @rec(pp::printer s,
               option::t[codemap] cm,
               option::t[vec[lexer::cmnt]] comments,
               option::t[vec[lexer::lit]] literals,
               mutable uint cur_cmnt,
               mutable uint cur_lit,
               mutable vec[pp::breaks] boxes,
               mode mode);

fn rust_printer(io::writer writer) -> ps {
    let vec[pp::breaks] boxes = [];
    ret @rec(s=pp::mk_printer(writer, default_columns),
             cm=none[codemap],
             comments=none[vec[lexer::cmnt]],
             literals=none[vec[lexer::lit]],
             mutable cur_cmnt=0u,
             mutable cur_lit=0u,
             mutable boxes=boxes,
             mode=mo_untyped);
}

fn to_str[T](&T t, fn(&ps s, &T s) f) -> str {
    auto writer = io::string_writer();
    auto s = rust_printer(writer.get_writer());
    f(s, t);
    eof(s.s);
    ret writer.get_str();
}

fn print_file(session sess, ast::_mod _mod, str filename, io::writer out,
              mode mode) {
    let vec[pp::breaks] boxes = [];
    auto r = lexer::gather_comments_and_literals(sess, filename);
    auto s = @rec(s=pp::mk_printer(out, default_columns),
                  cm=some(sess.get_codemap()),
                  comments=some(r.cmnts),
                  literals=some(r.lits),
                  mutable cur_cmnt=0u,
                  mutable cur_lit=0u,
                  mutable boxes = boxes,
                  mode=mode);
    print_mod(s, _mod);
    eof(s.s);
}

fn ty_to_str(&ast::ty ty) -> str { be to_str(ty, print_type); }
fn pat_to_str(&@ast::pat pat) -> str { be to_str(pat, print_pat); }
fn expr_to_str(&@ast::expr e) -> str { be to_str(e, print_expr); }
fn stmt_to_str(&ast::stmt s) -> str { be to_str(s, print_stmt); }
fn item_to_str(&@ast::item i) -> str { be to_str(i, print_item); }
fn path_to_str(&ast::path p) -> str { be to_str(p, print_path); }
fn lit_to_str(&@ast::lit l) -> str { be to_str(l, print_literal); }

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

fn ibox(&ps s, uint u) {
    vec::push(s.boxes, pp::inconsistent);
    pp::ibox(s.s, u);
}

fn cbox(&ps s, uint u) {
    vec::push(s.boxes, pp::consistent);
    pp::cbox(s.s, u);
}

fn box(&ps s, uint u, pp::breaks b) {
    vec::push(s.boxes, b);
    pp::box(s.s, u, b);
}

fn end(&ps s) {
    vec::pop(s.boxes);
    pp::end(s.s);
}


fn word_nbsp(&ps s, str w) {
    word(s.s, w);
    word(s.s, " ");
}

fn word_space(&ps s, str w) {
    word(s.s, w);
    space(s.s);
}

fn popen(&ps s) {
    word(s.s, "(");
}

fn pclose(&ps s) {
    word(s.s, ")");
}

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

// Synthesizes a comment that was not textually present in the original source
// file.
fn synth_comment(&ps s, str text) {
    word(s.s, "/*");
    space(s.s);
    word(s.s, text);
    space(s.s);
    word(s.s, "*/");
}

fn commasep[IN](&ps s, breaks b, vec[IN] elts, fn(&ps, &IN) op) {
    box(s, 0u, b);
    auto first = true;
    for (IN elt in elts) {
        if (first) {first = false;}
        else {word_space(s, ",");}
        op(s, elt);
    }
    end(s);
}

fn commasep_cmnt[IN](&ps s, breaks b, vec[IN] elts, fn(&ps, &IN) op,
                     fn(&IN) -> common::span get_span) {
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
            space(s.s);
        }
    }
    end(s);
}

fn commasep_exprs(&ps s, breaks b, vec[@ast::expr] exprs) {
    fn expr_span(&@ast::expr expr) -> common::span {ret expr.span;}
    auto f = print_expr;
    auto gs = expr_span;
    commasep_cmnt[@ast::expr](s, b, exprs, f, gs);
}

fn print_mod(&ps s, ast::_mod _mod) {
    for (@ast::view_item vitem in _mod.view_items) {
        print_view_item(s, vitem);
    }
    for (@ast::item item in _mod.items) {
        // Mod-level item printing we're a little more space-y about.
        hardbreak(s.s);
        print_item(s, item);
    }
    print_remaining_comments(s);
}

fn print_boxed_type(&ps s, &@ast::ty ty) { print_type(s, *ty); }
fn print_type(&ps s, &ast::ty ty) {

    maybe_print_comment(s, ty.span.lo);
    ibox(s, 0u);
    alt (ty.node) {
        case (ast::ty_nil) {word(s.s, "()");}
        case (ast::ty_bool) {word(s.s, "bool");}
        case (ast::ty_bot) {word(s.s, "!");}
        case (ast::ty_int) {word(s.s, "int");}
        case (ast::ty_uint) {word(s.s, "uint");}
        case (ast::ty_float) {word(s.s, "float");}
        case (ast::ty_machine(?tm)) {word(s.s, common::ty_mach_to_str(tm));}
        case (ast::ty_char) {word(s.s, "char");}
        case (ast::ty_str) {word(s.s, "str");}
        case (ast::ty_box(?mt)) {word(s.s, "@"); print_mt(s, mt);}
        case (ast::ty_vec(?mt)) {
            word(s.s, "vec["); print_mt(s, mt); word(s.s, "]");
        }
        case (ast::ty_port(?t)) {
            word(s.s, "port["); print_type(s, *t); word(s.s, "]");
        }
        case (ast::ty_chan(?t)) {
            word(s.s, "chan["); print_type(s, *t); word(s.s, "]");
        }
        case (ast::ty_type) {word(s.s, "type");}
        case (ast::ty_tup(?elts)) {
            word(s.s, "tup");
            popen(s);
            auto f = print_mt;
            commasep[ast::mt](s, inconsistent, elts, f);
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
            fn get_span(&ast::ty_field f) -> common::span {
                ret f.span;
            }
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::ty_field](s, consistent, fields, f, gs);
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
                            m.node.inputs, m.node.output, m.node.cf);
                word(s.s, ";");
                end(s);
            }
            bclose(s, ty.span);
        }
        case (ast::ty_fn(?proto,?inputs,?output,?cf)) {
            print_ty_fn(s, proto, none[str], inputs, output, cf);
        }
        case (ast::ty_path(?path,_)) {
            print_path(s, path);
        }
    }
    end(s);
}

fn print_item(&ps s, &@ast::item item) {

    hardbreak(s.s);
    maybe_print_comment(s, item.span.lo);
    alt (item.node) {
        case (ast::item_const(?id, ?ty, ?expr, _, _)) {
            head(s, "const");
            print_type(s, *ty);
            space(s.s);
            word_space(s, id);
            end(s); // end the head-ibox
            word_space(s, "=");
            print_expr(s, expr);
            word(s.s, ";");
            end(s); // end the outer cbox
        }
        case (ast::item_fn(?name,?_fn,?typarams,_,_)) {
            print_fn(s, _fn.decl, _fn.proto, name, typarams);
            word(s.s, " ");
            print_block(s, _fn.body);
        }
        case (ast::item_mod(?id,?_mod,_)) {
            head(s, "mod");
            word_nbsp(s, id);
            bopen(s);
            for (@ast::item itm in _mod.items) {print_item(s, itm);}
            bclose(s, item.span);
        }
        case (ast::item_native_mod(?id,?nmod,_)) {
            head(s, "native");
            alt (nmod.abi) {
                case (ast::native_abi_rust) {word_nbsp(s, "\"rust\"");}
                case (ast::native_abi_cdecl) {word_nbsp(s, "\"cdecl\"");}
                case (ast::native_abi_rust_intrinsic) {
                    word_nbsp(s, "\"rust-intrinsic\"");
                }
            }
            word_nbsp(s, "mod");
            word_nbsp(s, id);
            bopen(s);
            for (@ast::native_item item in nmod.items) {
                ibox(s, indent_unit);
                maybe_print_comment(s, item.span.lo);
                alt (item.node) {
                    case (ast::native_item_ty(?id,_)) {
                        word_nbsp(s, "type");
                        word(s.s, id);
                    }
                    case (ast::native_item_fn(?id,?lname,?decl,
                                             ?typarams,_,_)) {
                        print_fn(s, decl, ast::proto_fn, id, typarams);
                        end(s); // end head-ibox
                        alt (lname) {
                            case (none) {}
                            case (some(?ss)) {
                                print_string(s, ss);
                            }
                        }
                    }
                }
                word(s.s, ";");
                end(s);
            }
            bclose(s, item.span);
        }
        case (ast::item_ty(?id,?ty,?params,_,_)) {
            ibox(s, indent_unit);
            ibox(s, 0u);
            word_nbsp(s, "type");
            word(s.s, id);
            print_type_params(s, params);
            end(s); // end the inner ibox
            space(s.s);
            word_space(s, "=");
            print_type(s, *ty);
            word(s.s, ";");
            end(s); // end the outer ibox
            break_offset(s.s, 0u, 0);
        }
        case (ast::item_tag(?id,?variants,?params,_,_)) {
            head(s, "tag");
            word(s.s, id);
            print_type_params(s, params);
            space(s.s);
            bopen(s);
            for (ast::variant v in variants) {
                space(s.s);
                maybe_print_comment(s, v.span.lo);
                word(s.s, v.node.name);
                if (vec::len[ast::variant_arg](v.node.args) > 0u) {
                    popen(s);
                    fn print_variant_arg(&ps s, &ast::variant_arg arg) {
                        print_type(s, *arg.ty);
                    }
                    auto f = print_variant_arg;
                    commasep[ast::variant_arg](s, consistent, v.node.args, f);
                    pclose(s);
                }
                word(s.s, ";");
                maybe_print_trailing_comment(s, v.span, none[uint]);
            }
            bclose(s, item.span);
        }
        case (ast::item_obj(?id,?_obj,?params,_,_)) {
            head(s, "obj");
            word(s.s, id);
            print_type_params(s, params);
            popen(s);
            fn print_field(&ps s, &ast::obj_field field) {
                ibox(s, indent_unit);
                print_type(s, *field.ty);
                space(s.s);
                word(s.s, field.ident);
                end(s);
            }
            fn get_span(&ast::obj_field f) -> common::span {ret f.ty.span;}
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::obj_field](s, consistent, _obj.fields, f, gs);
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
                case (_) {}
            }
            bclose(s, item.span);
        }
    }

    // Print the node ID if necessary. TODO: type as well.
    alt (s.mode) {
        case (mo_identified) {
            space(s.s);
            synth_comment(s, uint::to_str(ty::item_ann(item).id, 10u));
        }
        case (_) { /* no-op */ }
    }
}

fn print_stmt(&ps s, &ast::stmt st) {
    maybe_print_comment(s, st.span.lo);
    alt (st.node) {
        case (ast::stmt_decl(?decl,_)) {
            print_decl(s, decl);
        }
        case (ast::stmt_expr(?expr,_)) {
            space(s.s);
            print_expr(s, expr);
        }
    }
    if (front::parser::stmt_ends_with_semi(st)) {word(s.s, ";");}
    maybe_print_trailing_comment(s, st.span, none[uint]);
}

fn print_block(&ps s, ast::block blk) {
    maybe_print_comment(s, blk.span.lo);
    bopen(s);
    for (@ast::stmt st in blk.node.stmts) {
        print_stmt(s, *st)

    }
    alt (blk.node.expr) {
        case (some(?expr)) {
            space(s.s);
            print_expr(s, expr);
            maybe_print_trailing_comment(s, expr.span,
                                         some(blk.span.hi));
        }
        case (_) {}
    }
    bclose(s, blk.span);

    // Print the node ID if necessary: TODO: type as well.
    alt (s.mode) {
        case (mo_identified) {
            space(s.s);
            synth_comment(s, "block " + uint::to_str(blk.node.a.id, 10u));
        }
        case (_) { /* no-op */ }
    }
}

fn next_lit(&ps s) -> option::t[lexer::lit] {
    alt (s.literals) {
        case (some(?lits)) {
            if (s.cur_lit < vec::len(lits)) {
                ret some(lits.(s.cur_lit));
            } else {ret none[lexer::lit];}
        }
        case (_) {ret none[lexer::lit];}
    }
}

fn print_literal(&ps s, &@ast::lit lit) {
    maybe_print_comment(s, lit.span.lo);

    alt (next_lit(s)) {
        case (some(?lt)) {
            if (lt.pos == lit.span.lo) {
                word(s.s, lt.lit);
                s.cur_lit += 1u;
                ret;
            }
        }
        case (_) {}
    }

    alt (lit.node) {
        case (ast::lit_str(?st)) {print_string(s, st);}
        case (ast::lit_char(?ch)) {
            word(s.s, "'" + escape_str(str::from_bytes([ch as u8]), '\'')
                + "'");
        }
        case (ast::lit_int(?val)) {
            word(s.s, common::istr(val));
        }
        case (ast::lit_uint(?val)) {
            word(s.s, common::uistr(val) + "u");
        }
        case (ast::lit_float(?fstr)) {
            word(s.s, fstr);
        }
        case (ast::lit_mach_int(?mach,?val)) {
            word(s.s, common::istr(val as int));
            word(s.s, common::ty_mach_to_str(mach));
        }
        case (ast::lit_mach_float(?mach,?val)) {
            // val is already a str
            word(s.s, val);
            word(s.s, common::ty_mach_to_str(mach));
        }
        case (ast::lit_nil) {word(s.s, "()");}
        case (ast::lit_bool(?val)) {
            if (val) {word(s.s, "true");} else {word(s.s, "false");}
        }
    }
}

fn print_expr(&ps s, &@ast::expr expr) {
    maybe_print_comment(s, expr.span.lo);
    ibox(s, indent_unit);

    alt (s.mode) {
        case (mo_untyped) { /* no-op */ }
        case (mo_typed(_)) { popen(s); }
        case (mo_identified) { popen(s); }
    }

    alt (expr.node) {
        case (ast::expr_vec(?exprs,?mut,_)) {
            ibox(s, indent_unit);
            word(s.s, "[");
            if (mut == ast::mut) {
                word_nbsp(s, "mutable");
            }
            commasep_exprs(s, inconsistent, exprs);
            word(s.s, "]");
            end(s);
        }
        case (ast::expr_tup(?exprs,_)) {
            fn printElt(&ps s, &ast::elt elt) {
                ibox(s, indent_unit);
                if (elt.mut == ast::mut) {word_nbsp(s, "mutable");}
                print_expr(s, elt.expr);
                end(s);
            }
            fn get_span(&ast::elt elt) -> common::span {ret elt.expr.span;}
            word(s.s, "tup");
            popen(s);
            auto f = printElt;
            auto gs = get_span;
            commasep_cmnt[ast::elt](s, inconsistent, exprs, f, gs);
            pclose(s);
        }
        case (ast::expr_rec(?fields,?wth,_)) {
            fn print_field(&ps s, &ast::field field) {
                ibox(s, indent_unit);
                if (field.node.mut == ast::mut) {word_nbsp(s, "mutable");}
                word(s.s, field.node.ident);
                word(s.s, "=");
                print_expr(s, field.node.expr);
                end(s);
            }
            fn get_span(&ast::field field) -> common::span {
                ret field.span;
            }
            word(s.s, "rec");
            popen(s);
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::field](s, consistent, fields, f, gs);
            alt (wth) {
                case (some(?expr)) {
                    if (vec::len[ast::field](fields) > 0u) {space(s.s);}
                    ibox(s, indent_unit);
                    word_space(s, "with");
                    print_expr(s, expr);
                    end(s);
                }
                case (_) {}
            }
            pclose(s);
        }
        case (ast::expr_call(?func,?args,_)) {
            print_expr(s, func);
            popen(s);
            commasep_exprs(s, inconsistent, args);
            pclose(s);
        }
        case (ast::expr_self_method(?ident,_)) {
            word(s.s, "self.");
            print_ident(s, ident);
        }
        case (ast::expr_bind(?func,?args,_)) {
            fn print_opt(&ps s, &option::t[@ast::expr] expr) {
                alt (expr) {
                    case (some(?expr)) {
                        print_expr(s, expr);
                    }
                    case (_) {word(s.s, "_");}
                }
            }
            word_nbsp(s, "bind");
            print_expr(s, func);
            popen(s);
            auto f = print_opt;
            commasep[option::t[@ast::expr]](s, inconsistent, args, f);
            pclose(s);
        }
    case (ast::expr_spawn(_,_,?e,?es,_)) {
          word_nbsp(s, "spawn");
          print_expr(s, e);
          popen(s);
          commasep_exprs(s, inconsistent, es);
          pclose(s);
        }
        case (ast::expr_binary(?op,?lhs,?rhs,_)) {
            auto prec = operator_prec(op);
            print_maybe_parens(s, lhs, prec);
            space(s.s);
            word_space(s, ast::binop_to_str(op));
            print_maybe_parens(s, rhs, prec + 1);
        }
        case (ast::expr_unary(?op,?expr,_)) {
            word(s.s, ast::unop_to_str(op));
            print_expr(s, expr);
        }
        case (ast::expr_lit(?lit,_)) {
            print_literal(s, lit);
        }
        case (ast::expr_cast(?expr,?ty,_)) {
            print_maybe_parens(s, expr, front::parser::as_prec);
            space(s.s);
            word_space(s, "as");
            print_type(s, *ty);
        }
        case (ast::expr_if(?test,?block,?elseopt,_)) {

            head(s, "if");
            popen(s);
            print_expr(s, test);
            pclose(s);
            space(s.s);
            print_block(s, block);
            fn do_else(&ps s, option::t[@ast::expr] els) {
                alt (els) {
                    case (some(?_else)) {
                        alt (_else.node) {
                            // "another else-if"
                            case (ast::expr_if(?i,?t,?e,_)) {
                                cbox(s, indent_unit-1u);
                                ibox(s, 0u);
                                word(s.s, " else if ");
                                popen(s);
                                print_expr(s, i);
                                pclose(s);
                                space(s.s);
                                print_block(s, t);
                                do_else(s, e);
                            }
                            // "final else"
                            case (ast::expr_block(?b, _)) {
                                cbox(s, indent_unit-1u);
                                ibox(s, 0u);
                                word(s.s, " else ");
                                print_block(s, b);
                            }
                        }
                    }
                    case (_) { /* fall through */ }
                }
            }
            do_else(s, elseopt);
        }
        case (ast::expr_while(?test,?block,_)) {
            head(s, "while");
            popen(s);
            print_expr(s, test);
            pclose(s);
            space(s.s);
            print_block(s, block);
        }
        case (ast::expr_for(?decl,?expr,?block,_)) {
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
        case (ast::expr_for_each(?decl,?expr,?block,_)) {
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
        case (ast::expr_do_while(?block,?expr,_)) {
            head(s, "do");
            space(s.s);
            print_block(s, block);
            space(s.s);
            word_space(s, "while");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_alt(?expr,?arms,_)) {
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
        case (ast::expr_block(?block,_)) {
            // containing cbox, will be closed by print-block at }
            cbox(s, indent_unit);

            // head-box, will be closed by print-block after {
            ibox(s, 0u);
            print_block(s, block);
        }
        case (ast::expr_move(?lhs,?rhs,_)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "<-");
            print_expr(s, rhs);
        }
        case (ast::expr_assign(?lhs,?rhs,_)) {
            print_expr(s, lhs);
            space(s.s);
            word_space(s, "=");
            print_expr(s, rhs);
        }
        case (ast::expr_assign_op(?op,?lhs,?rhs,_)) {
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
            print_expr(s, rhs);
            space(s.s);
            word_space(s, "|>");
            print_expr(s, lhs);
        }
        case (ast::expr_field(?expr,?id,_)) {
            print_expr(s, expr);
            word(s.s, ".");
            word(s.s, id);
        }
        case (ast::expr_index(?expr,?index,_)) {
            print_expr(s, expr);
            word(s.s, ".");
            popen(s);
            print_expr(s, index);
            pclose(s);
        }
        case (ast::expr_path(?path,_)) {
            print_path(s, path);
        }
        case (ast::expr_fail(_)) {
            word(s.s, "fail");
        }
        case (ast::expr_break(_)) {
            word(s.s, "break");
        }
        case (ast::expr_cont(_)) {
            word(s.s, "cont");
        }
        case (ast::expr_ret(?result,_)) {
            word(s.s, "ret");
            alt (result) {
                case (some(?expr)) {
                    word(s.s, " ");
                    print_expr(s, expr);
                }
                case (_) {}
            }
        }
        case (ast::expr_put(?result,_)) {
            word(s.s, "put");
            alt (result) {
                case (some(?expr)) {
                    word(s.s, " ");
                    print_expr(s, expr);
                }
                case (_) {}
            }
        }
        case (ast::expr_be(?result,_)) {
            word_nbsp(s, "be");
            print_expr(s, result);
        }
        case (ast::expr_log(?lvl,?expr,_)) {
            alt (lvl) {
                case (1) {word_nbsp(s, "log");}
                case (0) {word_nbsp(s, "log_err");}
            }
            print_expr(s, expr);
        }
        case (ast::expr_check(?expr,_)) {
            word_nbsp(s, "check");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_assert(?expr,_)) {
            word_nbsp(s, "assert");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }
        case (ast::expr_ext(?path, ?args, ?body, _, _)) {
            word(s.s, "#");
            print_path(s, path);
            if (vec::len[@ast::expr](args) > 0u) {
                popen(s);
                commasep_exprs(s, inconsistent, args);
                pclose(s);
            }
            // FIXME: extension 'body'
        }
        case (ast::expr_port(_)) {
            word(s.s, "port");
            popen(s);
            pclose(s);
        }
        case (ast::expr_chan(?expr, _)) {
            word(s.s, "chan");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }

        case (ast::expr_anon_obj(_,_,_,_)) {
            word(s.s, "anon obj");
            // TODO: nicer pretty-printing of anon objs
        }
    }

    // Print the type or node ID if necessary.
    alt (s.mode) {
        case (mo_untyped) { /* no-op */ }
        case (mo_typed(?tcx)) {
            space(s.s);
            word(s.s, "as");
            space(s.s);
            word(s.s, ty::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
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
            space(s.s);
            ibox(s, indent_unit);
            alt (loc.ty) {
                case (some(?ty)) {
                    word_nbsp(s, "let");
                    print_type(s, *ty);
                    space(s.s);
                }
                case (_) {
                    word_nbsp(s, "auto");

                    // Print the type or node ID if necessary.
                    alt (s.mode) {
                        case (mo_untyped) { /* no-op */ }
                        case (mo_typed(?tcx)) {
                            auto lty =
                                ty::ann_to_type(tcx, loc.ann);
                            word_space(s, ty::ty_to_str(tcx, lty));
                        }
                        case (mo_identified) { /* no-op */ }
                    }
                }
            }
            word(s.s, loc.ident);
            alt (loc.init) {
                case (some(?init)) {
                    space(s.s);
                    alt (init.op) {
                        case (ast::init_assign) {
                            word_space(s, "=");
                        }
                        case (ast::init_move) {
                            word_space(s, "<-");
                        }
                        case (ast::init_recv) {
                            word_space(s, "|>");
                        }
                    }
                    print_expr(s, init.expr);
                }
                case (_) {}
            }
            end(s);
        }
        case (ast::decl_item(?item)) {
            print_item(s, item);
        }
    }
}

fn print_ident(&ps s, &ast::ident ident) {
    word(s.s, ident);
}

fn print_for_decl(&ps s, @ast::decl decl) {
    alt (decl.node) {
        case (ast::decl_local(?loc)) {
            print_type(s, *option::get[@ast::ty](loc.ty));
            space(s.s);
            word(s.s, loc.ident);
        }
    }
}

fn print_path(&ps s, &ast::path path) {
    maybe_print_comment(s, path.span.lo);
    auto first = true;
    for (str id in path.node.idents) {
        if (first) {first = false;}
        else {word(s.s, "::");}
        word(s.s, id);
    }
    if (vec::len[@ast::ty](path.node.types) > 0u) {
        word(s.s, "[");
        auto f = print_boxed_type;
        commasep[@ast::ty](s, inconsistent, path.node.types, f);
        word(s.s, "]");
    }
}

fn print_pat(&ps s, &@ast::pat pat) {
    maybe_print_comment(s, pat.span.lo);
    alt (pat.node) {
        case (ast::pat_wild(_)) {word(s.s, "_");}
        case (ast::pat_bind(?id,_,_)) {word(s.s, "?" + id);}
        case (ast::pat_lit(?lit,_)) {print_literal(s, lit);}
        case (ast::pat_tag(?path,?args,_)) {
            print_path(s, path);
            if (vec::len[@ast::pat](args) > 0u) {
                popen(s);
                auto f = print_pat;
                commasep[@ast::pat](s, inconsistent, args, f);
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
        case (_) { /* no-op */ }
    }
}

fn print_fn(&ps s, ast::fn_decl decl, ast::proto proto, str name,
            vec[ast::ty_param] typarams) {
    alt (decl.purity) {
        case (ast::impure_fn) {
            if (proto == ast::proto_iter) {
                head(s, "iter");
            } else {
                head(s, "fn");
            }
        }
        case (_) {
            head(s, "pred");
        }
    }
    word(s.s, name);
    print_type_params(s, typarams);
    popen(s);
    fn print_arg(&ps s, &ast::arg x) {
        ibox(s, indent_unit);
        if (x.mode == ast::alias) {word(s.s, "&");}
        print_type(s, *x.ty);
        space(s.s);
        word(s.s, x.ident);
        end(s);
    }
    auto f = print_arg;
    commasep[ast::arg](s, inconsistent, decl.inputs, f);
    pclose(s);
    maybe_print_comment(s, decl.output.span.lo);
    if (decl.output.node != ast::ty_nil) {
        space(s.s);
        word_space(s, "->");
        print_type(s, *decl.output);
    }
}

fn print_type_params(&ps s, &vec[ast::ty_param] params) {
    if (vec::len[ast::ty_param](params) > 0u) {
        word(s.s, "[");
        fn printParam(&ps s, &ast::ty_param param) {
            word(s.s, param);
        }
        auto f = printParam;
        commasep[ast::ty_param](s, inconsistent, params, f);
        word(s.s, "]");
    }
}

fn print_view_item(&ps s, &@ast::view_item item) {
    hardbreak(s.s);
    maybe_print_comment(s, item.span.lo);
    alt (item.node) {
        case (ast::view_item_use(?id,?mta,_,_)) {
            head(s, "use");
            word(s.s, id);
            if (vec::len[@ast::meta_item](mta) > 0u) {
                popen(s);
                fn print_meta(&ps s, &@ast::meta_item item) {
                    ibox(s, indent_unit);
                    word_space(s, item.node.name);
                    word_space(s, "=");
                    print_string(s, item.node.value);
                    end(s);
                }
                auto f = print_meta;
                commasep[@ast::meta_item](s, consistent, mta, f);
                pclose(s);
            }
        }
        case (ast::view_item_import(?id,?ids,_)) {
            head(s, "import");
            if (!str::eq(id, ids.(vec::len[str](ids)-1u))) {
                word_space(s, id);
                word_space(s, "=");
            }
            auto first = true;
            for (str elt in ids) {
                if (first) {first = false;}
                else {word(s.s, "::");}
                word(s.s, elt);
            }
        }
        case (ast::view_item_import_glob(?ids,_)) {
            head(s, "import");
            auto first = true;
            for (str elt in ids) {
                if (first) {first = false;}
                else {word(s.s, "::");}
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
        if (spec.op == op) {ret spec.prec;}
    }
    fail;
}

fn print_maybe_parens(&ps s, &@ast::expr expr, int outer_prec) {
    auto add_them;
    alt (expr.node) {
        case (ast::expr_binary(?op,_,_,_)) {
            add_them = operator_prec(op) < outer_prec;
        }
        case (ast::expr_cast(_,_,_)) {
            add_them = front::parser::as_prec < outer_prec;
        }
        case (_) {
            add_them = false;
        }
    }
    if (add_them) {popen(s);}
    print_expr(s, expr);
    if (add_them) {pclose(s);}
}

fn escape_str(str st, char to_escape) -> str {
    let str out = "";
    auto len = str::byte_len(st);
    auto i = 0u;
    while (i < len) {
        alt (st.(i) as char) {
            case ('\n') {out += "\\n";}
            case ('\t') {out += "\\t";}
            case ('\r') {out += "\\r";}
            case ('\\') {out += "\\\\";}
            case (?cur) {
                if (cur == to_escape) {out += "\\";}
                // FIXME some (or all?) non-ascii things should be escaped
                str::push_char(out, cur);
            }
        }
        i += 1u;
    }
    ret out;
}

fn print_mt(&ps s, &ast::mt mt) {
    alt (mt.mut) {
        case (ast::mut)       { word_nbsp(s, "mutable");  }
        case (ast::maybe_mut) { word_nbsp(s, "mutable?"); }
        case (ast::imm)       { /* nothing */        }
    }
    print_type(s, *mt.ty);
}

fn print_string(&ps s, &str st) {
    word(s.s, "\""); word(s.s, escape_str(st, '"')); word(s.s, "\"");
}

fn print_ty_fn(&ps s, &ast::proto proto, &option::t[str] id,
               &vec[ast::ty_arg] inputs, &@ast::ty output,
               &ast::controlflow cf) {
    ibox(s, indent_unit);
    if (proto == ast::proto_fn) {word(s.s, "fn");}
    else {word(s.s, "iter");}
    alt (id) {
        case (some(?id)) {
            word(s.s, " ");
            word(s.s, id);
        }
        case (_) {}
    }
    zerobreak(s.s);
    popen(s);
    fn print_arg(&ps s, &ast::ty_arg input) {
        if (input.node.mode == ast::alias) {word(s.s, "&");}
        print_type(s, *input.node.ty);
    }
    auto f = print_arg;
    commasep[ast::ty_arg](s, inconsistent, inputs, f);
    pclose(s);
    maybe_print_comment(s, output.span.lo);
    if (output.node != ast::ty_nil) {
        space(s.s);
        ibox(s, indent_unit);
        word_space(s, "->");
        alt (cf) {
            case (ast::return) {
                print_type(s, *output);
            }
            case (ast::noreturn) {
                word_nbsp(s, "!");
            }
        }
        end(s);
    }
    end(s);
}

fn next_comment(&ps s) -> option::t[lexer::cmnt] {
    alt (s.comments) {
        case (some(?cmnts)) {
            if (s.cur_cmnt < vec::len(cmnts)) {
                ret some(cmnts.(s.cur_cmnt));
            } else {ret none[lexer::cmnt];}
        }
        case (_) {ret none[lexer::cmnt];}
    }
}

fn maybe_print_comment(&ps s, uint pos) {
    while (true) {
        alt (next_comment(s)) {
            case (some(?cmnt)) {
                if (cmnt.pos < pos) {
                    print_comment(s, cmnt);
                    s.cur_cmnt += 1u;
                } else { break; }
            }
            case (_) {break;}
        }
    }
}

fn maybe_print_trailing_comment(&ps s, common::span span,
                                option::t[uint] next_pos) {
    auto cm;
    alt (s.cm) {
        case (some(?ccm)) {
            cm = ccm;
        }
        case (_) { ret; }
    }
    alt (next_comment(s)) {
        case (some(?cmnt)) {
            if (cmnt.style != lexer::trailing) { ret; }

            auto span_line = codemap::lookup_pos(cm, span.hi);
            auto comment_line = codemap::lookup_pos(cm, cmnt.pos);
            auto next = cmnt.pos + 1u;
            alt (next_pos) {
                case (none) { }
                case (some(?p)) { next = p; }
            }
            if (span.hi < cmnt.pos &&
                cmnt.pos < next &&
                span_line.line == comment_line.line) {
                print_comment(s, cmnt);
                s.cur_cmnt += 1u;
            }
        }
        case (_) {}
    }
}

fn print_remaining_comments(&ps s) {
    while (true) {
        alt (next_comment(s)) {
            case (some(?cmnt)) {
                print_comment(s, cmnt);
                s.cur_cmnt += 1u;
            }
            case (_) {break;}
        }
    }
}

fn in_cbox(&ps s) -> bool {
    auto len = vec::len(s.boxes);
    if (len == 0u) { ret false; }
    ret s.boxes.(len-1u) == pp::consistent;
}

fn print_comment(&ps s, lexer::cmnt cmnt) {
    alt (cmnt.style) {
        case (lexer::mixed) {
            assert vec::len(cmnt.lines) == 1u;
            zerobreak(s.s);
            word(s.s, cmnt.lines.(0));
            zerobreak(s.s);
        }

        case (lexer::isolated) {
            hardbreak(s.s);
            ibox(s, 0u);
            for (str line in cmnt.lines) {
                word(s.s, line);
                hardbreak(s.s);
            }
            end(s);
        }

        case (lexer::trailing) {
            word(s.s, " ");
            if (vec::len(cmnt.lines) == 1u) {
                word(s.s, cmnt.lines.(0));
                hardbreak(s.s);
            } else {
                ibox(s, 0u);
                for (str line in cmnt.lines) {
                    word(s.s, line);
                    hardbreak(s.s);
                }
                end(s);
            }
        }
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
