import std::vec;
import std::str;
import std::io;
import std::option;
import driver::session::session;
import front::ast;
import front::lexer;
import middle::ty;
import util::common;
import pp;

import pp::printer;
import pp::break_offset;
import pp::cbox;
import pp::ibox;
import pp::wrd;
import pp::space;
import pp::hardbreak;
import pp::end;
import pp::eof;

const uint indent_unit = 4u;
const uint default_columns = 78u;

tag mode {
    mo_untyped;
    mo_typed(ty::ctxt);
}

type ps = @rec(pp::printer s,
               option::t[vec[lexer::cmnt]] comments,
               mutable uint cur_cmnt,
               mode mode);

fn print_file(session sess, ast::_mod _mod, str filename, io::writer out,
              mode mode) {
    auto cmnts = lexer::gather_comments(sess, filename);
    auto s = @rec(s=pp::mk_printer(out, default_columns),
                  comments=option::some[vec[lexer::cmnt]](cmnts),
                  mutable cur_cmnt=0u,
                  mode=mode);
    print_mod(s, _mod);
    eof(s.s);
}

fn ty_to_str(&@ast::ty ty) -> str {
    auto writer = io::string_writer();
    auto s = @rec(s=pp::mk_printer(writer.get_writer(), default_columns),
                  comments=option::none[vec[lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
    print_type(s, ty);
    eof(s.s);
    ret writer.get_str();
}

fn block_to_str(&ast::block blk) -> str {
    auto writer = io::string_writer();
    auto s = @rec(s=pp::mk_printer(writer.get_writer(), default_columns),
                  comments=option::none[vec[lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
    cbox(s.s, indent_unit); // containing cbox, will be closed by print-block at }
    ibox(s.s, 0u); // head-ibox, will be closed by print-block after {
    print_block(s, blk);
    eof(s.s);
    ret writer.get_str();
}

fn pat_to_str(&@ast::pat p) -> str {
    auto writer = io::string_writer();
    auto s = @rec(s=pp::mk_printer(writer.get_writer(), default_columns),
                  comments=option::none[vec[lexer::cmnt]],
                  mutable cur_cmnt=0u,
                  mode=mo_untyped);
    print_pat(s, p);
    eof(s.s);
    ret writer.get_str();
}

fn word_nbsp(ps s, str word) {
    wrd(s.s, word);
    wrd(s.s, " ");
}

fn word_space(ps s, str word) {
    wrd(s.s, word);
    space(s.s);
}

fn popen(ps s) {
    wrd(s.s, "(");
}

fn pclose(ps s) {
    wrd(s.s, ")");
}

fn head(ps s, str word) {
    // outer-box is consistent
    cbox(s.s, indent_unit);
    // head-box is inconsistent
    ibox(s.s, str::char_len(word) + 1u);
    // keyword that starts the head
    word_nbsp(s, word);
}

fn bopen(ps s) {
    wrd(s.s, "{");
    end(s.s); // close the head-box
}

fn bclose(ps s, common::span span) {
    maybe_print_comment(s, span.hi);
    break_offset(s.s, 1u, -(indent_unit as int));
    wrd(s.s, "}");
    end(s.s); // close the outer-box
}

fn commasep[IN](ps s, vec[IN] elts, fn(ps, &IN) op) {
    ibox(s.s, 0u);
    auto first = true;
    for (IN elt in elts) {
        if (first) {first = false;}
        else {word_space(s, ",");}
        op(s, elt);
    }
    end(s.s);
}

fn commasep_cmnt[IN](ps s, vec[IN] elts, fn(ps, &IN) op,
                     fn(&IN) -> common::span get_span) {
    ibox(s.s, 0u);
    auto len = vec::len[IN](elts);
    auto i = 0u;
    for (IN elt in elts) {
        op(s, elt);
        i += 1u;
        if (i < len) {
            wrd(s.s, ",");
            if (!maybe_print_line_comment(s, get_span(elt))) {space(s.s);}
        }
    }
    end(s.s);
}

fn commasep_exprs(ps s, vec[@ast::expr] exprs) {
    fn expr_span(&@ast::expr expr) -> common::span {ret expr.span;}
    auto f = print_expr;
    auto gs = expr_span;
    commasep_cmnt[@ast::expr](s, exprs, f, gs);
}

fn print_mod(ps s, ast::_mod _mod) {
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

fn print_type(ps s, &@ast::ty ty) {

    maybe_print_comment(s, ty.span.lo);
    ibox(s.s, 0u);
    alt (ty.node) {
        case (ast::ty_nil) {wrd(s.s, "()");}
        case (ast::ty_bool) {wrd(s.s, "bool");}
        case (ast::ty_bot) {wrd(s.s, "_|_");}
        case (ast::ty_int) {wrd(s.s, "int");}
        case (ast::ty_uint) {wrd(s.s, "uint");}
        case (ast::ty_float) {wrd(s.s, "float");}
        case (ast::ty_machine(?tm)) {wrd(s.s, common::ty_mach_to_str(tm));}
        case (ast::ty_char) {wrd(s.s, "char");}
        case (ast::ty_str) {wrd(s.s, "str");}
        case (ast::ty_box(?mt)) {wrd(s.s, "@"); print_mt(s, mt);}
        case (ast::ty_vec(?mt)) {
            wrd(s.s, "vec["); print_mt(s, mt); wrd(s.s, "]");
        }
        case (ast::ty_port(?t)) {
            wrd(s.s, "port["); print_type(s, t); wrd(s.s, "]");
        }
        case (ast::ty_chan(?t)) {
            wrd(s.s, "chan["); print_type(s, t); wrd(s.s, "]");
        }
        case (ast::ty_type) {wrd(s.s, "type");}
        case (ast::ty_tup(?elts)) {
            wrd(s.s, "tup");
            popen(s);
            auto f = print_mt;
            commasep[ast::mt](s, elts, f);
            pclose(s);
        }
        case (ast::ty_rec(?fields)) {
            wrd(s.s, "rec");
            popen(s);
            fn print_field(ps s, &ast::ty_field f) {
                cbox(s.s, indent_unit);
                print_mt(s, f.mt);
                space(s.s);
                wrd(s.s, f.ident);
                end(s.s);
            }
            fn get_span(&ast::ty_field f) -> common::span {
              // Try to reconstruct the span for this field
              auto sp = f.mt.ty.span;
              auto hi = sp.hi + str::char_len(f.ident) + 1u;
              ret rec(hi=hi with sp);
            }
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::ty_field](s, fields, f, gs);
            pclose(s);
        }
        case (ast::ty_obj(?methods)) {
            head(s, "obj");
            bopen(s);
            for (ast::ty_method m in methods) {
                cbox(s.s, indent_unit);
                print_ty_fn(s, m.proto, option::some[str](m.ident),
                            m.inputs, m.output, m.cf);
                wrd(s.s, ";");
                end(s.s);
            }
            bclose(s, ty.span);
        }
        case (ast::ty_fn(?proto,?inputs,?output,?cf)) {
            print_ty_fn(s, proto, option::none[str], inputs, output, cf);
        }
        case (ast::ty_path(?path,_)) {
            print_path(s, path);
        }
    }
    end(s.s);
}

fn print_item(ps s, @ast::item item) {

    hardbreak(s.s);
    maybe_print_comment(s, item.span.lo);
    alt (item.node) {
        case (ast::item_const(?id, ?ty, ?expr, _, _)) {
            head(s, "const");
            print_type(s, ty);
            space(s.s);
            word_space(s, id);
            end(s.s); // end the head-ibox
            word_space(s, "=");
            print_expr(s, expr);
            wrd(s.s, ";");
            end(s.s); // end the outer cbox
        }
        case (ast::item_fn(?name,?_fn,?typarams,_,_)) {
            print_fn(s, _fn.decl, name, typarams);
            wrd(s.s, " ");
            print_block(s, _fn.body);
        }
        case (ast::item_mod(?id,?_mod,_)) {
            head(s, "mod");
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
                ibox(s.s, indent_unit);
                maybe_print_comment(s, item.span.lo);
                alt (item.node) {
                    case (ast::native_item_ty(?id,_)) {
                        word_nbsp(s, "type");
                        wrd(s.s, id);
                    }
                    case (ast::native_item_fn(?id,?lname,?decl,
                                             ?typarams,_,_)) {
                        print_fn(s, decl, id, typarams);
                        end(s.s); // end head-ibox
                        alt (lname) {
                            case (option::none[str]) {}
                            case (option::some[str](?ss)) {
                                print_string(s, ss);
                            }
                        }
                    }
                }
                wrd(s.s, ";");
                end(s.s);
            }
            bclose(s, item.span);
        }
        case (ast::item_ty(?id,?ty,?params,_,_)) {
            ibox(s.s, indent_unit);
            ibox(s.s, 0u);
            word_nbsp(s, "type");
            wrd(s.s, id);
            print_type_params(s, params);
            end(s.s); // end the inner ibox
            space(s.s);
            word_space(s, "=");
            print_type(s, ty);
            wrd(s.s, ";");
            end(s.s); // end the outer ibox
            break_offset(s.s, 0u, 0);
        }
        case (ast::item_tag(?id,?variants,?params,_,_)) {
            head(s, "tag");
            wrd(s.s, id);
            print_type_params(s, params);
            space(s.s);
            bopen(s);
            for (ast::variant v in variants) {
                space(s.s);
                maybe_print_comment(s, v.span.lo);
                wrd(s.s, v.node.name);
                if (vec::len[ast::variant_arg](v.node.args) > 0u) {
                    popen(s);
                    fn print_variant_arg(ps s, &ast::variant_arg arg) {
                        print_type(s, arg.ty);
                    }
                    auto f = print_variant_arg;
                    commasep[ast::variant_arg](s, v.node.args, f);
                    pclose(s);
                }
                wrd(s.s, ";");
                maybe_print_line_comment(s, v.span);
            }
            bclose(s, item.span);
        }
        case (ast::item_obj(?id,?_obj,?params,_,_)) {
            head(s, "obj");
            wrd(s.s, id);
            print_type_params(s, params);
            popen(s);
            fn print_field(ps s, &ast::obj_field field) {
                ibox(s.s, indent_unit);
                print_type(s, field.ty);
                space(s.s);
                wrd(s.s, field.ident);
                end(s.s);
            }
            fn get_span(&ast::obj_field f) -> common::span {ret f.ty.span;}
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::obj_field](s, _obj.fields, f, gs);
            pclose(s);
            space(s.s);
            bopen(s);
            for (@ast::method meth in _obj.methods) {
                let vec[ast::ty_param] typarams = [];
                hardbreak(s.s);
                maybe_print_comment(s, meth.span.lo);
                print_fn(s, meth.node.meth.decl, meth.node.ident, typarams);
                wrd(s.s, " ");
                print_block(s, meth.node.meth.body);
            }
            alt (_obj.dtor) {
                case (option::some[@ast::method](?dtor)) {
                    head(s, "drop");
                    print_block(s, dtor.node.meth.body);
                }
                case (_) {}
            }
            bclose(s, item.span);
        }
    }
}

fn print_block(ps s, ast::block blk) {
    maybe_print_comment(s, blk.span.lo);
    bopen(s);
    auto first = true;
    for (@ast::stmt st in blk.node.stmts) {
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
        if (front::parser::stmt_ends_with_semi(st)) {wrd(s.s, ";");}
        maybe_print_line_comment(s, st.span);
    }
    alt (blk.node.expr) {
        case (option::some[@ast::expr](?expr)) {
            space(s.s);
            print_expr(s, expr);
            maybe_print_line_comment(s, expr.span);
        }
        case (_) {}
    }
    bclose(s, blk.span);
}

fn print_literal(ps s, @ast::lit lit) {
    maybe_print_comment(s, lit.span.lo);
    alt (lit.node) {
        case (ast::lit_str(?st)) {print_string(s, st);}
        case (ast::lit_char(?ch)) {
            wrd(s.s, "'" + escape_str(str::from_bytes([ch as u8]), '\'')
                + "'");
        }
        case (ast::lit_int(?val)) {
            wrd(s.s, common::istr(val));
        }
        case (ast::lit_uint(?val)) { // FIXME clipping? uistr?
            wrd(s.s, common::istr(val as int) + "u");
        }
        case (ast::lit_float(?fstr)) {
            wrd(s.s, fstr);
        }
        case (ast::lit_mach_int(?mach,?val)) {
            wrd(s.s, common::istr(val as int));
            wrd(s.s, common::ty_mach_to_str(mach));
        }
        case (ast::lit_mach_float(?mach,?val)) {
            // val is already a str
            wrd(s.s, val);
            wrd(s.s, common::ty_mach_to_str(mach));
        }
        case (ast::lit_nil) {wrd(s.s, "()");}
        case (ast::lit_bool(?val)) {
            if (val) {wrd(s.s, "true");} else {wrd(s.s, "false");}
        }
    }
}

fn print_expr(ps s, &@ast::expr expr) {
    maybe_print_comment(s, expr.span.lo);
    ibox(s.s, indent_unit);

    alt (s.mode) {
        case (mo_untyped) { /* no-op */ }
        case (mo_typed(_)) { popen(s); }
    }

    alt (expr.node) {
        case (ast::expr_vec(?exprs,?mut,_)) {
            if (mut == ast::mut) {
                word_nbsp(s, "mutable");
            }
            ibox(s.s, indent_unit);
            wrd(s.s, "[");
            commasep_exprs(s, exprs);
            wrd(s.s, "]");
            end(s.s);
        }
        case (ast::expr_tup(?exprs,_)) {
            fn printElt(ps s, &ast::elt elt) {
                ibox(s.s, indent_unit);
                if (elt.mut == ast::mut) {word_nbsp(s, "mutable");}
                print_expr(s, elt.expr);
                end(s.s);
            }
            fn get_span(&ast::elt elt) -> common::span {ret elt.expr.span;}
            wrd(s.s, "tup");
            popen(s);
            auto f = printElt;
            auto gs = get_span;
            commasep_cmnt[ast::elt](s, exprs, f, gs);
            pclose(s);
        }
        case (ast::expr_rec(?fields,?wth,_)) {
            fn print_field(ps s, &ast::field field) {
                ibox(s.s, indent_unit);
                if (field.mut == ast::mut) {word_nbsp(s, "mutable");}
                wrd(s.s, field.ident);
                wrd(s.s, "=");
                print_expr(s, field.expr);
                end(s.s);
            }
            fn get_span(&ast::field field) -> common::span {
                ret field.expr.span;
            }
            wrd(s.s, "rec");
            popen(s);
            auto f = print_field;
            auto gs = get_span;
            commasep_cmnt[ast::field](s, fields, f, gs);
            alt (wth) {
                case (option::some[@ast::expr](?expr)) {
                    if (vec::len[ast::field](fields) > 0u) {space(s.s);}
                    ibox(s.s, indent_unit);
                    word_space(s, "with");
                    print_expr(s, expr);
                    end(s.s);
                }
                case (_) {}
            }
            pclose(s);
        }
        case (ast::expr_call(?func,?args,_)) {
            print_expr(s, func);
            popen(s);
            commasep_exprs(s, args);
            pclose(s);
        }
        case (ast::expr_self_method(?ident,_)) {
            wrd(s.s, "self.");
            print_ident(s, ident);
        }
        case (ast::expr_bind(?func,?args,_)) {
            fn print_opt(ps s, &option::t[@ast::expr] expr) {
                alt (expr) {
                    case (option::some[@ast::expr](?expr)) {
                        print_expr(s, expr);
                    }
                    case (_) {wrd(s.s, "_");}
                }
            }
            word_nbsp(s, "bind");
            print_expr(s, func);
            popen(s);
            auto f = print_opt;
            commasep[option::t[@ast::expr]](s, args, f);
            pclose(s);
        }
    case (ast::expr_spawn(_,_,?e,?es,_)) {
          word_nbsp(s, "spawn");
          print_expr(s, e);
          popen(s);
          commasep_exprs(s, es);
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
            wrd(s.s, ast::unop_to_str(op));
            print_expr(s, expr);
        }
        case (ast::expr_lit(?lit,_)) {
            print_literal(s, lit);
        }
        case (ast::expr_cast(?expr,?ty,_)) {
            print_maybe_parens(s, expr, front::parser::as_prec);
            space(s.s);
            word_space(s, "as");
            print_type(s, ty);
        }
        case (ast::expr_if(?test,?block,?elseopt,_)) {
            head(s, "if");
            popen(s);
            print_expr(s, test);
            pclose(s);
            space(s.s);
            print_block(s, block);
            alt (elseopt) {
                case (option::some[@ast::expr](?_else)) {
                    // NB: we can't use 'head' here since
                    // it builds a block that starts in the
                    // wrong column.
                    cbox(s.s, indent_unit-1u);
                    ibox(s.s, 0u);
                    wrd(s.s, " else ");
                    alt (_else.node) {
                        case (ast::expr_block(?b, _)) {
                            print_block(s, block);
                        }
                    }
                }
                case (_) { /* fall through */ }
            }
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
            cbox(s.s, indent_unit); // containing cbox, will be closed by print-block at }
            ibox(s.s, 0u); // head-box, will be closed by print-block after {
            print_block(s, block);
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
            wrd(s.s, ast::binop_to_str(op));
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
            wrd(s.s, ".");
            wrd(s.s, id);
        }
        case (ast::expr_index(?expr,?index,_)) {
            print_expr(s, expr);
            wrd(s.s, ".");
            popen(s);
            print_expr(s, index);
            pclose(s);
        }
        case (ast::expr_path(?path,_)) {
            print_path(s, path);
        }
        case (ast::expr_fail(_)) {
            wrd(s.s, "fail");
        }
        case (ast::expr_break(_)) {
            wrd(s.s, "break");
        }
        case (ast::expr_cont(_)) {
            wrd(s.s, "cont");
        }
        case (ast::expr_ret(?result,_)) {
            wrd(s.s, "ret");
            alt (result) {
                case (option::some[@ast::expr](?expr)) {
                    wrd(s.s, " ");
                    print_expr(s, expr);
                }
                case (_) {}
            }
        }
        case (ast::expr_put(?result,_)) {
            wrd(s.s, "put");
            alt (result) {
                case (option::some[@ast::expr](?expr)) {
                    wrd(s.s, " ");
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
            wrd(s.s, "#");
            print_path(s, path);
            if (vec::len[@ast::expr](args) > 0u) {
                popen(s);
                commasep_exprs(s, args);
                pclose(s);
            }
            // FIXME: extension 'body'
        }
        case (ast::expr_port(_)) {
            wrd(s.s, "port");
            popen(s);
            pclose(s);
        }
        case (ast::expr_chan(?expr, _)) {
            wrd(s.s, "chan");
            popen(s);
            print_expr(s, expr);
            pclose(s);
        }

        case (ast::expr_anon_obj(_,_,_,_)) {
            wrd(s.s, "anon obj");
            // TODO: nicer pretty-printing of anon objs
        }
    }

    // Print the type if necessary.
    alt (s.mode) {
        case (mo_untyped) { /* no-op */ }
        case (mo_typed(?tcx)) {
            space(s.s);
            wrd(s.s, "as");
            space(s.s);
            wrd(s.s, ty::ty_to_str(tcx, ty::expr_ty(tcx, expr)));
            pclose(s);
        }
    }

    end(s.s);
}

fn print_decl(ps s, @ast::decl decl) {
    maybe_print_comment(s, decl.span.lo);
    alt (decl.node) {
        case (ast::decl_local(?loc)) {
            space(s.s);
            ibox(s.s, indent_unit);
            alt (loc.ty) {
                case (option::some[@ast::ty](?ty)) {
                    word_nbsp(s, "let");
                    print_type(s, ty);
                    space(s.s);
                }
                case (_) {
                    word_nbsp(s, "auto");

                    // Print the type if necessary.
                    alt (s.mode) {
                        case (mo_untyped) { /* no-op */ }
                        case (mo_typed(?tcx)) {
                            auto lty =
                                ty::ann_to_type(tcx.node_types, loc.ann);
                            word_space(s, ty::ty_to_str(tcx, lty));
                        }
                    }
                }
            }
            wrd(s.s, loc.ident);
            alt (loc.init) {
                case (option::some[ast::initializer](?init)) {
                    space(s.s);
                    alt (init.op) {
                        case (ast::init_assign) {
                            word_space(s, "=");
                        }
                        case (ast::init_recv) {
                            word_space(s, "<-");
                        }
                    }
                    print_expr(s, init.expr);
                }
                case (_) {}
            }
            end(s.s);
        }
        case (ast::decl_item(?item)) {
            print_item(s, item);
        }
    }
}

fn print_ident(ps s, ast::ident ident) {
    wrd(s.s, ident);
}

fn print_for_decl(ps s, @ast::decl decl) {
    alt (decl.node) {
        case (ast::decl_local(?loc)) {
            print_type(s, option::get[@ast::ty](loc.ty));
            space(s.s);
            wrd(s.s, loc.ident);
        }
    }
}

fn print_path(ps s, ast::path path) {
    maybe_print_comment(s, path.span.lo);
    auto first = true;
    for (str id in path.node.idents) {
        if (first) {first = false;}
        else {wrd(s.s, "::");}
        wrd(s.s, id);
    }
    if (vec::len[@ast::ty](path.node.types) > 0u) {
        wrd(s.s, "[");
        auto f = print_type;
        commasep[@ast::ty](s, path.node.types, f);
        wrd(s.s, "]");
    }
}

fn print_pat(ps s, &@ast::pat pat) {
    maybe_print_comment(s, pat.span.lo);
    alt (pat.node) {
        case (ast::pat_wild(_)) {wrd(s.s, "_");}
        case (ast::pat_bind(?id,_,_)) {wrd(s.s, "?" + id);}
        case (ast::pat_lit(?lit,_)) {print_literal(s, lit);}
        case (ast::pat_tag(?path,?args,_)) {
            print_path(s, path);
            if (vec::len[@ast::pat](args) > 0u) {
                popen(s);
                auto f = print_pat;
                commasep[@ast::pat](s, args, f);
                pclose(s);
            }
        }
    }
}

fn print_fn(ps s, ast::fn_decl decl, str name,
                   vec[ast::ty_param] typarams) {
    alt (decl.purity) {
        case (ast::impure_fn) {
            head(s, "fn");
        }
        case (_) {
            head(s, "pred");
        }
    }
    wrd(s.s, name);
    print_type_params(s, typarams);
    popen(s);
    fn print_arg(ps s, &ast::arg x) {
        ibox(s.s, indent_unit);
        if (x.mode == ast::alias) {wrd(s.s, "&");}
        print_type(s, x.ty);
        space(s.s);
        wrd(s.s, x.ident);
        end(s.s);
    }
    auto f = print_arg;
    commasep[ast::arg](s, decl.inputs, f);
    pclose(s);
    maybe_print_comment(s, decl.output.span.lo);
    if (decl.output.node != ast::ty_nil) {
        space(s.s);
        word_space(s, "->");
        print_type(s, decl.output);
    }
}

fn print_type_params(ps s, vec[ast::ty_param] params) {
    if (vec::len[ast::ty_param](params) > 0u) {
        wrd(s.s, "[");
        fn printParam(ps s, &ast::ty_param param) {
            wrd(s.s, param);
        }
        auto f = printParam;
        commasep[ast::ty_param](s, params, f);
        wrd(s.s, "]");
    }
}

fn print_view_item(ps s, @ast::view_item item) {
    hardbreak(s.s);
    maybe_print_comment(s, item.span.lo);
    alt (item.node) {
        case (ast::view_item_use(?id,?mta,_,_)) {
            head(s, "use");
            wrd(s.s, id);
            if (vec::len[@ast::meta_item](mta) > 0u) {
                popen(s);
                fn print_meta(ps s, &@ast::meta_item item) {
                    ibox(s.s, indent_unit);
                    word_space(s, item.node.name);
                    word_space(s, "=");
                    print_string(s, item.node.value);
                    end(s.s);
                }
                auto f = print_meta;
                commasep[@ast::meta_item](s, mta, f);
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
                else {wrd(s.s, "::");}
                wrd(s.s, elt);
            }
        }
        case (ast::view_item_export(?id)) {
            head(s, "export");
            wrd(s.s, id);
        }
    }
    wrd(s.s, ";");
    end(s.s); // end inner head-block
    end(s.s); // end outer head-block
}

// FIXME: The fact that this builds up the table anew for every call is
// not good. Eventually, table should be a const.
fn operator_prec(ast::binop op) -> int {
    for (front::parser::op_spec spec in front::parser::prec_table()) {
        if (spec.op == op) {ret spec.prec;}
    }
    fail;
}

fn print_maybe_parens(ps s, @ast::expr expr, int outer_prec) {
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

fn print_mt(ps s, &ast::mt mt) {
    alt (mt.mut) {
        case (ast::mut)       { word_nbsp(s, "mutable");  }
        case (ast::maybe_mut) { word_nbsp(s, "mutable?"); }
        case (ast::imm)       { /* nothing */        }
    }
    print_type(s, mt.ty);
}

fn print_string(ps s, str st) {
    wrd(s.s, "\""); wrd(s.s, escape_str(st, '"')); wrd(s.s, "\"");
}

fn print_ty_fn(ps s, ast::proto proto, option::t[str] id,
               vec[ast::ty_arg] inputs, @ast::ty output,
               ast::controlflow cf) {
    if (proto == ast::proto_fn) {wrd(s.s, "fn");}
    else {wrd(s.s, "iter");}
    alt (id) {
        case (option::some[str](?id)) {space(s.s); wrd(s.s, id);}
        case (_) {}
    }
    popen(s);
    fn print_arg(ps s, &ast::ty_arg input) {
        if (input.mode == ast::alias) {wrd(s.s, "&");}
        print_type(s, input.ty);
    }
    auto f = print_arg;
    commasep[ast::ty_arg](s, inputs, f);
    pclose(s);
    maybe_print_comment(s, output.span.lo);
    if (output.node != ast::ty_nil) {
        space(s.s);
        ibox(s.s, indent_unit);
        word_space(s, "->");
        alt (cf) {
            case (ast::return) {
                print_type(s, output);
            }
            case (ast::noreturn) {
                word_nbsp(s, "!");
            }
        }
        end(s.s);
    }
}

fn next_comment(ps s) -> option::t[lexer::cmnt] {
    alt (s.comments) {
        case (option::some[vec[lexer::cmnt]](?cmnts)) {
            if (s.cur_cmnt < vec::len[lexer::cmnt](cmnts)) {
                ret option::some[lexer::cmnt](cmnts.(s.cur_cmnt));
            } else {ret option::none[lexer::cmnt];}
        }
        case (_) {ret option::none[lexer::cmnt];}
    }
}

fn maybe_print_comment(ps s, uint pos) {
    auto first = true;
    while (true) {
        alt (next_comment(s)) {
            case (option::some[lexer::cmnt](?cmnt)) {
                if (cmnt.pos < pos) {
                    if (first) {
                        first = false;
                        break_offset(s.s, 0u, 0);
                    }
                    print_comment(s, cmnt.val);
                    s.cur_cmnt += 1u;
                } else { break; }
            }
            case (_) {break;}
        }
    }
}

fn maybe_print_line_comment(ps s, common::span span) -> bool {
    alt (next_comment(s)) {
        case (option::some[lexer::cmnt](?cmnt)) {
            if (span.hi + 4u >= cmnt.pos) {
                wrd(s.s, " ");
                print_comment(s, cmnt.val);
                s.cur_cmnt += 1u;
                ret true;
            }
        }
        case (_) {}
    }
    ret false;
}

fn print_remaining_comments(ps s) {
    auto first = true;
    while (true) {
        alt (next_comment(s)) {
            case (option::some[lexer::cmnt](?cmnt)) {
                if (first) {
                    first = false;
                    break_offset(s.s, 0u, 0);
                }
                print_comment(s, cmnt.val);
                s.cur_cmnt += 1u;
            }
            case (_) {break;}
        }
    }
}

fn print_comment(ps s, lexer::cmnt_ cmnt) {
    alt (cmnt) {
        case (lexer::cmnt_line(?val)) {
            wrd(s.s, "// " + val);
            hardbreak(s.s);
        }
        case (lexer::cmnt_block(?lines)) {
            cbox(s.s, 1u);
            wrd(s.s, "/*");
            auto first = true;
            for (str ln in lines) {
                if (first) {
                    first = false;
                } else {
                    hardbreak(s.s);
                }
                wrd(s.s, ln);
            }
            wrd(s.s, "*/");
            end(s.s);
            hardbreak(s.s);
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
