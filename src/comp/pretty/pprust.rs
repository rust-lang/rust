import std._vec;
import std._str;
import std.option;
import front.ast;
import pp.box; import pp.abox; import pp.vbox;
import pp.end; import pp.wrd; import pp.space; import pp.line;
import pp.ps;

import foo = std.io;

const uint indent_unit = 2u;
const int as_prec = 5;

impure fn print_ast(ast._mod _mod, std.io.writer out) {
  auto s = pp.mkstate(out, 80u);
  for (@ast.view_item vitem in _mod.view_items) {print_view_item(s, vitem);}
  line(s);
  for (@ast.item item in _mod.items) {print_item(s, item);}
}

fn ty_to_str(&@ast.ty ty) -> str {
  auto writer = std.io.string_writer();
  print_type(pp.mkstate(writer.get_writer(), 0u), ty);
  ret writer.get_str();
}

impure fn hbox(ps s) {
  pp.hbox(s, indent_unit);
}
impure fn wrd1(ps s, str word) {
  wrd(s, word);
  space(s);
}
impure fn popen(ps s) {
  wrd(s, "(");
  abox(s);
}
impure fn pclose(ps s) {
  end(s);
  wrd(s, ")");
}
impure fn bopen(ps s) {
  wrd1(s, "{");
  vbox(s, indent_unit);
  line(s);
}
impure fn bclose(ps s) {
  end(s);
  pp.cwrd(s, "}");
}
impure fn commasep[IN](ps s, vec[IN] elts, impure fn (ps, IN) op) {
  auto first = true;
  for (IN elt in elts) {
    if (first) {first = false;}
    else {wrd1(s, ",");}
    op(s, elt);
  }
}

impure fn print_type(ps s, @ast.ty ty) {
  hbox(s);
  alt (ty.node) {
    case (ast.ty_nil) {wrd(s, "()");}
    case (ast.ty_bool) {wrd(s, "bool");}
    case (ast.ty_int) {wrd(s, "int");}
    case (ast.ty_uint) {wrd(s, "uint");}
    case (ast.ty_machine(?tm)) {wrd(s, util.common.ty_mach_to_str(tm));}
    case (ast.ty_char) {wrd(s, "char");}
    case (ast.ty_str) {wrd(s, "str");}
    case (ast.ty_box(?t)) {wrd(s, "@"); print_type(s, t);}
    case (ast.ty_vec(?t)) {wrd(s, "vec["); print_type(s, t); wrd(s, "]");}
    case (ast.ty_type) {wrd(s, "type");}
    case (ast.ty_tup(?elts)) {
      wrd(s, "tup");
      popen(s);
      auto f = print_type;
      commasep[@ast.ty](s, elts, f);
      pclose(s);
    }
    case (ast.ty_rec(?fields)) {
      wrd(s, "rec");
      popen(s);
      impure fn print_field(ps s, ast.ty_field f) {
        hbox(s);
        print_type(s, f.ty);
        space(s);
        wrd(s, f.ident);
        end(s);
      }
      auto f = print_field;
      commasep[ast.ty_field](s, fields, f);
      pclose(s);
    }
    case (ast.ty_obj(?methods)) {
      wrd1(s, "obj");
      bopen(s);
      for (ast.ty_method m in methods) {
        hbox(s);
        print_ty_fn(s, m.proto, option.some[str](m.ident),
                    m.inputs, m.output);
        wrd(s, ";");
        end(s);
        line(s);
      }
      bclose(s);
    }
    case (ast.ty_fn(?proto,?inputs,?output)) {
      print_ty_fn(s, proto, option.none[str], inputs, output);
    }
    case (ast.ty_path(?path,_)) {
      print_path(s, path);
    }
    case (ast.ty_mutable(?t)) {
      wrd1(s, "mutable");
      print_type(s, t);
    }
  }
  end(s);
}

impure fn print_item(ps s, @ast.item item) {
  hbox(s);
  alt (item.node) {
    case (ast.item_const(?id, ?ty, ?expr, _, _)) {
      wrd1(s, "const");
      print_type(s, ty);
      space(s);
      wrd1(s, id);
      wrd1(s, "=");
      print_expr(s, expr);
      wrd(s, ";");
    }
    case (ast.item_fn(?name,?_fn,?typarams,_,_)) {
      print_fn(s, _fn.decl, name, typarams);
      space(s);
      print_block(s, _fn.body);
    }
    case (ast.item_mod(?id,?_mod,_)) {
      wrd1(s, "mod");
      wrd1(s, id);
      bopen(s);
      for (@ast.item itm in _mod.items) {print_item(s, itm);}
      bclose(s);
    }
    case (ast.item_native_mod(?id,?nmod,_)) {
      wrd1(s, "native");
      alt (nmod.abi) {
        case (ast.native_abi_rust) {wrd1(s, "\"rust\"");}
        case (ast.native_abi_cdecl) {wrd1(s, "\"cdecl\"");}
      }
      wrd1(s, "mod");
      wrd1(s, id);
      bopen(s);
      for (@ast.native_item item in nmod.items) {
        hbox(s);
        alt (item.node) {
          case (ast.native_item_ty(?id,_)) {
            wrd1(s, "type");
            wrd(s, id);
          }
          case (ast.native_item_fn(?id,?decl,?typarams,_,_)) {
            print_fn(s, decl, id, typarams);
          }
        }
        wrd(s, ";");
        end(s);
      }
      bclose(s);
    }
    case (ast.item_ty(?id,?ty,?params,_,_)) {
      wrd1(s, "type");
      wrd(s, id);
      print_type_params(s, params);
      space(s);
      wrd1(s, "=");
      print_type(s, ty);
      wrd(s, ";");
    }
    case (ast.item_tag(?id,?variants,?params,_)) {
      wrd1(s, "tag");
      wrd(s, id);
      print_type_params(s, params);
      space(s);
      bopen(s);
      for (ast.variant v in variants) {
        wrd(s, v.name);
        if (_vec.len[ast.variant_arg](v.args) > 0u) {
          popen(s);
          impure fn print_variant_arg(ps s, ast.variant_arg arg) {
            print_type(s, arg.ty);
          }
          auto f = print_variant_arg;
          commasep[ast.variant_arg](s, v.args, f);
          pclose(s);
        }
        wrd(s, ";");
        line(s);
      }
      bclose(s);
    }
    case (ast.item_obj(?id,?_obj,?params,_,_)) {
      wrd1(s, "obj");
      wrd(s, id);
      print_type_params(s, params);
      popen(s);
      impure fn print_field(ps s, ast.obj_field field) {
        hbox(s);
        print_type(s, field.ty);
        space(s);
        wrd(s, field.ident);
        end(s);
      }
      auto f = print_field;
      commasep[ast.obj_field](s, _obj.fields, f);
      pclose(s);
      space(s);
      bopen(s);
      for (@ast.method meth in _obj.methods) {
        hbox(s);
        let vec[ast.ty_param] typarams = vec();
        print_fn(s, meth.node.meth.decl, meth.node.ident, typarams);
        space(s);
        print_block(s, meth.node.meth.body);
        end(s);
        line(s);
      }
      alt (_obj.dtor) {
        case (option.some[ast.block](?dtor)) {
          hbox(s);
          wrd1(s, "close");
          print_block(s, dtor);
          end(s);
          line(s);
        }
        case (_) {}
      }
      bclose(s);
    }
  }
  end(s);
  line(s);
  line(s);
}

impure fn print_block(ps s, ast.block blk) {
  bopen(s);
  for (@ast.stmt st in blk.node.stmts) {
    alt (st.node) {
      case (ast.stmt_decl(?decl)) {print_decl(s, decl);}
      case (ast.stmt_expr(?expr)) {print_expr(s, expr);}
    }
    if (front.parser.stmt_ends_with_semi(st)) {wrd(s, ";");}
    line(s);
  }
  alt (blk.node.expr) {
    case (option.some[@ast.expr](?expr)) {
      print_expr(s, expr);
      line(s);
    }
    case (_) {}
  }
  bclose(s);
}

impure fn print_literal(ps s, @ast.lit lit) {
  alt (lit.node) {
    case (ast.lit_str(?st)) {print_string(s, st);}
    case (ast.lit_char(?ch)) {
      wrd(s, "'" + escape_str(_str.from_bytes(vec(ch as u8)), '\'') + "'");
    }
    case (ast.lit_int(?val)) {
      wrd(s, util.common.istr(val));
    }
    case (ast.lit_uint(?val)) { // TODO clipping? uistr?
      wrd(s, util.common.istr(val as int) + "u");
    }
    case (ast.lit_mach_int(?mach,?val)) {
      wrd(s, util.common.istr(val as int));
      wrd(s, util.common.ty_mach_to_str(mach));
    }
    case (ast.lit_nil) {wrd(s, "()");}
    case (ast.lit_bool(?val)) {
      if (val) {wrd(s, "true");} else {wrd(s, "false");}
    }
  }
}

impure fn print_expr(ps s, @ast.expr expr) {
  auto pe = print_expr;
  hbox(s);
  alt (expr.node) {
    case (ast.expr_vec(?exprs,_)) {
      wrd(s, "vec");
      popen(s);
      commasep[@ast.expr](s, exprs, pe);
      pclose(s);
    }
    case (ast.expr_tup(?exprs,_)) {
      impure fn printElt(ps s, ast.elt elt) {
        hbox(s);
        if (elt.mut == ast.mut) {wrd1(s, "mutable");}
        print_expr(s, elt.expr);
        end(s);
      }
      wrd(s, "tup");
      popen(s);
      auto f = printElt;
      commasep[ast.elt](s, exprs, f);
      pclose(s);
    }
    case (ast.expr_rec(?fields,_,_)) {
      impure fn print_field(ps s, ast.field field) {
        hbox(s);
        if (field.mut == ast.mut) {wrd1(s, "mutable");}
        wrd(s, field.ident);
        wrd(s, "=");
        print_expr(s, field.expr);
        end(s);
      }
      wrd(s, "rec");
      popen(s);
      auto f = print_field;
      commasep[ast.field](s, fields, f);
      pclose(s);
    }
    case (ast.expr_call(?func,?args,_)) {
      print_expr(s, func);
      popen(s);
      commasep[@ast.expr](s, args, pe);
      pclose(s);
    }
    case (ast.expr_bind(?func,?args,_)) {
      impure fn print_opt(ps s, option.t[@ast.expr] expr) {
        alt (expr) {
          case (option.some[@ast.expr](?expr)) {
            print_expr(s, expr);
          }
          case (_) {wrd(s, "_");}
        }
      }
      wrd1(s, "bind");
      print_expr(s, func);
      popen(s);
      auto f = print_opt;
      commasep[option.t[@ast.expr]](s, args, f);
      pclose(s);
    }
    case (ast.expr_binary(?op,?lhs,?rhs,_)) {
      auto prec = operator_prec(op);
      print_maybe_parens(s, lhs, prec);
      space(s);
      wrd1(s, ast.binop_to_str(op));
      print_maybe_parens(s, rhs, prec + 1);
    }
    case (ast.expr_unary(?op,?expr,_)) {
      wrd(s, ast.unop_to_str(op));
      if (op == ast._mutable) {space(s);}
      print_expr(s, expr);
    }
    case (ast.expr_lit(?lit,_)) {
      print_literal(s, lit);
    }
    case (ast.expr_cast(?expr,?ty,_)) {
      print_maybe_parens(s, expr, as_prec);
      space(s);
      wrd1(s, "as");
      print_type(s, ty);
    }
    case (ast.expr_if(?test,?block,?clauses,?_else,_)) {
      impure fn print_clause(ps s, @ast.expr test, ast.block blk) {
        wrd1(s, "if");
        popen(s);
        print_expr(s, test);
        pclose(s);
        space(s);
        print_block(s, blk);
      }
      print_clause(s, test, block);
      for (tup(@ast.expr, ast.block) clause in clauses) {
        space(s);
        wrd1(s, "else");
        print_clause(s, clause._0, clause._1);
      }
      alt (_else) {
        case (option.some[ast.block](?blk)) {
          space(s);
          wrd1(s, "else");
          print_block(s, blk);
        }
        case (_) { /* fall through */ }
      }
    }
    case (ast.expr_while(?test,?block,_)) {
      wrd1(s, "while");
      popen(s);
      print_expr(s, test);
      pclose(s);
      space(s);
      print_block(s, block);
    }
    case (ast.expr_for(?decl,?expr,?block,_)) {
      wrd1(s, "for");
      popen(s);
      print_decl(s, decl);
      space(s);
      wrd1(s, "in");
      print_expr(s, expr);
      pclose(s);
      space(s);
      print_block(s, block);
    }
    case (ast.expr_for_each(?decl,?expr,?block,_)) {
      wrd1(s, "for each");
      popen(s);
      print_decl(s, decl);
      space(s);
      wrd1(s, "in");
      print_expr(s, expr);
      space(s);
      print_block(s, block);
    }
    case (ast.expr_do_while(?block,?expr,_)) {
      wrd1(s, "do");
      space(s);
      print_block(s, block);
      space(s);
      wrd1(s, "while");
      popen(s);
      print_expr(s, expr);
      pclose(s);
    }
    case (ast.expr_alt(?expr,?arms,_)) {
      wrd1(s, "alt");
      popen(s);
      print_expr(s, expr);
      pclose(s);
      space(s);
      bopen(s);
      for (ast.arm arm in arms) {
        hbox(s);
        wrd1(s, "case");
        popen(s);
        print_pat(s, arm.pat);
        pclose(s);
        space(s);
        print_block(s, arm.block);
        end(s);
        line(s);
      }
      bclose(s);
    }
    case (ast.expr_block(?block,_)) {
      print_block(s, block);
    }
    case (ast.expr_assign(?lhs,?rhs,_)) {
      print_expr(s, lhs);
      space(s);
      wrd1(s, "=");
      print_expr(s, rhs);
    }
    case (ast.expr_assign_op(?op,?lhs,?rhs,_)) {
      print_expr(s, lhs);
      space(s);
      wrd(s, ast.binop_to_str(op));
      wrd1(s, "=");
      print_expr(s, rhs);
    }
    case (ast.expr_field(?expr,?id,_)) {
      print_expr(s, expr);
      wrd(s, ".");
      wrd(s, id);
    }
    case (ast.expr_index(?expr,?index,_)) {
      print_expr(s, expr);
      wrd(s, ".");
      popen(s);
      print_expr(s, index);
      pclose(s);
    }
    case (ast.expr_path(?path,_,_)) {
      print_path(s, path);
    }
    case (ast.expr_fail) {
      wrd(s, "fail");
    }
    case (ast.expr_ret(?result)) {
      wrd(s, "ret");
      alt (result) {
        case (option.some[@ast.expr](?expr)) {
          space(s);
          print_expr(s, expr);
        }
        case (_) {}
      }
    }
    case (ast.expr_put(?result)) {
      wrd(s, "put");
      alt (result) {
        case (option.some[@ast.expr](?expr)) {
          space(s);
          print_expr(s, expr);
        }
        case (_) {}
      }
    }
    case (ast.expr_be(?result)) {
      wrd1(s, "be");
      print_expr(s, result);
    }
    case (ast.expr_log(?expr)) {
      wrd1(s, "log");
      print_expr(s, expr);
    }
    case (ast.expr_check_expr(?expr)) {
      wrd1(s, "check");
      print_expr(s, expr);
    }
    case (ast.expr_ext(?path, ?args, ?body, _, _)) {
      wrd(s, "#");
      print_path(s, path);
      if (_vec.len[@ast.expr](args) > 0u) {
        popen(s);
        commasep[@ast.expr](s, args, pe);
        pclose(s);
      }
      // TODO: extension 'body'
    }
  }
  end(s);
}

impure fn print_decl(ps s, @ast.decl decl) {
  hbox(s);
  alt (decl.node) {
    case (ast.decl_local(?loc)) {
      alt (loc.ty) {
        case (option.some[@ast.ty](?ty)) {
          wrd1(s, "let");
          print_type(s, ty);
          space(s);
        }
        case (_) {
          wrd1(s, "auto");
        }
      }
      wrd(s, loc.ident);
      alt (loc.init) {
        case (option.some[@ast.expr](?init)) {
          space(s);
          wrd1(s, "=");
          print_expr(s, init);
        }
        case (_) {}
      }
    }
    case (ast.decl_item(?item)) {
      print_item(s, item);
    }
  }
  end(s);
}

impure fn print_path(ps s, ast.path path) {
  auto first = true;
  for (str id in path.node.idents) {
    if (first) {first = false;}
    else {wrd(s, ".");}
    wrd(s, id);
  }
  if (_vec.len[@ast.ty](path.node.types) > 0u) {
    wrd(s, "[");
    auto f = print_type;
    commasep[@ast.ty](s, path.node.types, f);
    wrd(s, "]");
  }
}

impure fn print_pat(ps s, @ast.pat pat) {
  alt (pat.node) {
    case (ast.pat_wild(_)) {wrd(s, "_");}
    case (ast.pat_bind(?id,_,_)) {wrd(s, "?" + id);}
    case (ast.pat_lit(?lit,_)) {print_literal(s, lit);}
    case (ast.pat_tag(?path,?args,_,_)) {
      print_path(s, path);
      if (_vec.len[@ast.pat](args) > 0u) {
        popen(s);
        auto f = print_pat;
        commasep[@ast.pat](s, args, f);
        pclose(s);
      }
    }
  }
}

impure fn print_fn(ps s, ast.fn_decl decl, str name,
                   vec[ast.ty_param] typarams) {
  alt (decl.effect) {
    case (ast.eff_impure) {wrd1(s, "impure");}
    case (ast.eff_unsafe) {wrd1(s, "unsafe");}
    case (_) {}
  }
  wrd1(s, "fn");
  wrd(s, name);
  print_type_params(s, typarams);
  popen(s);
  impure fn print_arg(ps s, ast.arg x) {
    hbox(s);
    print_type(s, x.ty);
    space(s);
    wrd(s, x.ident);
    end(s);
  }
  auto f = print_arg;
  commasep[ast.arg](s, decl.inputs, f);
  pclose(s);
  if (decl.output.node != ast.ty_nil) {
    space(s);
    hbox(s);
    wrd1(s, "->");
    print_type(s, decl.output);
    end(s);
  }
}

impure fn print_type_params(ps s, vec[ast.ty_param] params) {
  if (_vec.len[ast.ty_param](params) > 0u) {
    wrd(s, "[");
    impure fn printParam(ps s, ast.ty_param param) {wrd(s, param.ident);}
    auto f = printParam;
    commasep[ast.ty_param](s, params, f);
    wrd(s, "]");
  }
}

impure fn print_view_item(ps s, @ast.view_item item) {
  hbox(s);
  alt (item.node) {
    case (ast.view_item_use(?id,?mta,_)) {
      wrd1(s, "use");
      wrd(s, id);
      if (_vec.len[@ast.meta_item](mta) > 0u) {
        popen(s);
        impure fn print_meta(ps s, @ast.meta_item item) {
          hbox(s);
          wrd1(s, item.node.name);
          wrd1(s, "=");
          print_string(s, item.node.value);
          end(s);
        }
        auto f = print_meta;
        commasep[@ast.meta_item](s, mta, f);
        pclose(s);
      }
    }
    case (ast.view_item_import(?id,?ids,_,_)) {
      wrd1(s, "import");
      if (!_str.eq(id, ids.(_vec.len[str](ids)-1u))) {
        wrd1(s, id);
        wrd1(s, "=");
      }
      auto first = true;
      for (str elt in ids) {
        if (first) {first = false;}
        else {wrd(s, ".");}
        wrd(s, elt);
      }
    }
    case (ast.view_item_export(?id)) {
      wrd1(s, "export");
      wrd(s, id);
    }
  }
  end(s);
  wrd(s, ";");
  line(s);
}

// FIXME: The fact that this builds up the table anew for every call is
// not good. Eventually, table should be a const.
fn operator_prec(ast.binop op) -> int {
  for (front.parser.op_spec spec in front.parser.prec_table()) {
    if (spec.op == op) {ret spec.prec;}
  }
  fail;
}

impure fn print_maybe_parens(ps s, @ast.expr expr, int outer_prec) {
  auto add_them;
  alt (expr.node) {
    case (ast.expr_binary(?op,_,_,_)) {
      add_them = operator_prec(op) < outer_prec;
    }
    case (ast.expr_cast(_,_,_)) {
      add_them = as_prec < outer_prec;
    }
    case (_) {
      add_them = false;
    }
  }
  if (add_them) {popen(s);}
  print_expr(s, expr);
  if (add_them) {pclose(s);}
}

// TODO non-ascii
fn escape_str(str st, char to_escape) -> str {
  let str out = "";
  auto len = _str.byte_len(st);
  auto i = 0u;
  while (i < len) {
    alt (st.(i) as char) {
      case ('\n') {out += "\\n";}
      case ('\t') {out += "\\t";}
      case ('\r') {out += "\\r";}
      case ('\\') {out += "\\\\";}
      case (?cur) {
        if (cur == to_escape) {out += "\\";}
        out += cur as u8;
      }
    }
    i += 1u;
  }
  ret out;
}

impure fn print_string(ps s, str st) {
  wrd(s, "\""); wrd(s, escape_str(st, '"')); wrd(s, "\"");
}

impure fn print_ty_fn(ps s, ast.proto proto, option.t[str] id,
                      vec[ast.ty_arg] inputs, @ast.ty output) {
  if (proto == ast.proto_fn) {wrd(s, "fn");}
  else {wrd(s, "iter");}
  alt (id) {
    case (option.some[str](?id)) {space(s); wrd(s, id);}
    case (_) {}
  }
  popen(s);
  impure fn print_arg(ps s, ast.ty_arg input) {
    if (middle.ty.mode_is_alias(input.mode)) {wrd(s, "&");}
    print_type(s, input.ty);
  }
  auto f = print_arg;
  commasep[ast.ty_arg](s, inputs, f);
  pclose(s);
  if (output.node != ast.ty_nil) {
    space(s);
    hbox(s);
    wrd1(s, "->");
    print_type(s, output);
    end(s);
  }
}
