import rustc::driver::{driver,session};
import rustc::syntax::{ast, codemap};
import rustc::syntax::parse::parser;
import rustc::driver::diagnostic;
import rustc::syntax::print::pprust;
import rustc::syntax::codemap::span;
import rustc::middle::ty;
import rustc::middle::ast_map;
import rustc::util::ppaux;
import std::map::{hashmap, map, new_int_hash};
import std::getopts;
import io::writer_util;
import driver::build_session_options;
import driver::build_session;
import driver::build_configuration;

type parse_result = {
    crate: @ast::crate,
    tcx: ty::ctxt,
    roots: [str]
};

fn parse(argv: [str]) -> parse_result {
    let argv = argv;
    let argv0 = vec::shift(argv);
    let match = result::get(getopts::getopts(argv, driver::opts()));
    let sessopts = build_session_options(match, diagnostic::emit);
    let sess = build_session(sessopts, "", diagnostic::emit);
    let (ifile, roots) = {
        let free = match.free;
        if check vec::is_not_empty(free) {
            let t = vec::tail(free);
            (free[0], t)
        } else {
            fail "No input filename given.";
        }
    };
    let cfg = build_configuration(sess, argv0, ifile);
    alt driver::compile_upto(sess, cfg, ifile, driver::cu_typeck, none) {
      {crate, tcx: some(tcx)} { {crate:crate, tcx:tcx, roots:roots} }
      _ { fail "no type context"; }
    }
}

type ast_expr = str;
type ast_stmt = str;
type ast_blk = str;
type ast_pat = str;
type ast_ty = str;
type ast_item = str;

type tp_map = hashmap<ast::node_id, ty::t>;

type serialize_ctx = {
    crate: @ast::crate,
    tcx: ty::ctxt,

    serialize_tyfns: hashmap<ty::t, str>,
    deserialize_tyfns: hashmap<ty::t, str>,
    mutable item_fns: [ast_item],
    mutable constants: [str]

    // needed for #ast:
    // opts: @{cfg: ast::crate_cfg},
    // psess: parser::parse_sess
};

fn item_has_name(&&item: @ast::item, &&name: str) -> bool {
    item.ident == name
}

fn lookup(_mod: ast::_mod, idx: uint, names: [str]) -> @ast::item {
    let name = names[idx];
    alt vec::find(_mod.items, bind item_has_name(_, name)) {
      none { fail #fmt["cannot resolve name %s at index %u", name, idx]; }
      some(item) if idx == vec::len(names) - 1u { item }
      some(item) {
        alt item.node {
          ast::item_mod(_m) { lookup(_m, idx + 1u, names) }
          _ { fail #fmt["name %s at index %u not a module", name, idx]; }
        }
      }
    }
}

impl serialize_ctx for serialize_ctx {
    fn add_item(item: ast_item) {
        self.item_fns += [item];
    }

    fn tp_map(ty_params: [ast::ty_param], tps: [ty::t]) -> tp_map {
        assert vec::len(tps) == vec::len(ty_params);
        let tps_map = new_int_hash();
        vec::iter2(ty_params, tps) {|tp_def,tp_val|
            tps_map.insert(tp_def.id, tp_val);
        }
        ret tps_map;
    }

    fn ident(base_path: ast_map::path, id: str) -> str {
        #fmt["%s_%s", ast_map::path_to_str_with_sep(base_path, "_"), id]
    }

    fn instantiate(id: ast::def_id, args: [ty::t]) -> ty::t {
        let {bounds, ty} = ty::lookup_item_type(self.tcx, id);

        // typeck should guarantee this
        assert vec::len(*bounds) == vec::len(args);

        ret if vec::len(args) == 0u {
            ty
        } else {
            ty::substitute_type_params(self.tcx, args, ty)
        };
    }

    fn memoize(map: hashmap<ty::t, str>, base_name: str,
               ty0: ty::t, mk_fn: fn(str)) -> str {
        // check for existing function
        alt map.find(ty0) {
          some(name) { ret name; }
          none { /* fallthrough */ }
        }

        // define the name and insert into the hashtable
        // in case of recursive calls:
        let id = map.size();
        let name = #fmt["%s_%u", base_name, id];
        map.insert(ty0, name);
        mk_fn(name);
        ret name;
    }

    fn exec_named_item_fn(name: str, f: fn(ty::t) -> str) -> str {
        let names = str::split_str(name, "::");
        let item = lookup(self.crate.node.module, 0u, names);
        let def_id = {crate: ast::local_crate, node: item.id};
        let item_ty = self.instantiate(def_id, []);
        f(item_ty)
    }
}

impl serialize_methods for serialize_ctx {
    // fn session() -> parser::parse_sess { self.psess }

    fn mk_serialize_named_item_fn(name: str) -> str {
        self.exec_named_item_fn(name) {|item_ty|
            let fname = self.mk_serialize_ty_fn(item_ty);

            let ty_str = ppaux::ty_to_str(self.tcx, item_ty);
            check str::is_not_empty("::");
            let namep = str::replace(name, "::", "_");
            let item = #fmt["fn serialize_%s\
                                 <S:std::serialization::serializer>\n\
                                 (s: S, v: %s) {\n\
                                   %s(s, v);\n\
                                 }", namep, ty_str, fname];
            self.add_item(item);

            fname
        }
    }

    fn blk(stmts: [ast_stmt]) -> ast_blk {
        if vec::is_empty(stmts) {
            ""
        } else {
            "{" + str::connect(stmts, ";") + "}"
        }
    }

    fn blk_expr(stmts: [ast_stmt]) -> ast_expr {
        self.blk(stmts)
    }

    // Generates a function to serialize the given type.
    // Returns an AST fragment that names this function.
    fn serialize_ty(ty0: ty::t, v: ast_expr) -> ast_expr {
        let fname = self.mk_serialize_ty_fn(ty0);
        let ty0_str = ppaux::ty_to_str(self.tcx, ty0);
        #fmt["/*%s*/ %s(s, %s)", ty0_str, fname, v]
    }

    fn mk_serialize_ty_fn(ty0: ty::t) -> str {
        self.memoize(self.serialize_tyfns, "serialize", ty0) {|name|
            self.mk_serialize_ty_fn0(ty0, name)
        }
    }

    fn mk_serialize_ty_fn0(ty0: ty::t, name: str) {
        let ty0_str = ppaux::ty_to_str(self.tcx, ty0);
        let v = "v";

        let body_node = alt ty::get(ty0).struct {
          ty::ty_nil | ty::ty_bot { "()" }

          ty::ty_int(ast::ty_i)   { #fmt["\ns.emit_int(%s)\n", v] }
          ty::ty_int(ast::ty_i64) { #fmt["\ns.emit_i64(%s)\n", v] }
          ty::ty_int(ast::ty_i32) { #fmt["\ns.emit_i32(%s)\n", v] }
          ty::ty_int(ast::ty_i16) { #fmt["\ns.emit_i16(%s)\n", v] }
          ty::ty_int(ast::ty_i8)  { #fmt["\ns.emit_i8(%s)\n", v]  }

          ty::ty_int(ast::ty_char) { #fmt["\ns.emit_i8(%s as i8)\n", v] }

          ty::ty_uint(ast::ty_u)   { #fmt["\ns.emit_uint(%s)\n", v] }
          ty::ty_uint(ast::ty_u64) { #fmt["\ns.emit_u64(%s)\n", v] }
          ty::ty_uint(ast::ty_u32) { #fmt["\ns.emit_u32(%s)\n", v] }
          ty::ty_uint(ast::ty_u16) { #fmt["\ns.emit_u16(%s)\n", v] }
          ty::ty_uint(ast::ty_u8)  { #fmt["\ns.emit_u8(%s)\n", v]  }

          ty::ty_float(ast::ty_f64) { #fmt["\ns.emit_f64(%s)\n", v] }
          ty::ty_float(ast::ty_f32) { #fmt["\ns.emit_f32(%s)\n", v] }
          ty::ty_float(ast::ty_f)   { #fmt["\ns.emit_float(%s)\n", v] }

          ty::ty_bool { #fmt["\ns.emit_bool(%s)\n", v] }

          ty::ty_str { #fmt["\ns.emit_str(%s)\n", v] }

          ty::ty_enum(def_id, tps) { self.serialize_enum(v, def_id, tps) }
          ty::ty_box(mt) {
            let s = self.serialize_ty(mt.ty, #fmt["\n*%s\n", v]);
            #fmt["\ns.emit_box({||%s})\n", s]
          }
          ty::ty_uniq(mt) {
            let s = self.serialize_ty(mt.ty, #fmt["\n*%s\n", v]);
            #fmt["\ns.emit_uniq({||%s})\n", s]
          }
          ty::ty_vec(mt) {
            let selem = self.serialize_ty(mt.ty, "e");
            #fmt["\ns.emit_vec(vec::len(v), {||\n\
                    vec::iteri(v, {|i, e|\n\
                      s.emit_vec_elt(i, {||\n\
                          %s\n\
                  })})})\n", selem]
          }
          ty::ty_class(_, _) {
            fail "TODO--implement class";
          }
          ty::ty_rec(fields) {
            let stmts = vec::from_fn(vec::len(fields)) {|i|
                let field = fields[i];
                let f_name = field.ident;
                let f_ty = field.mt.ty;
                let efld = self.serialize_ty(f_ty, #fmt["\n%s.%s\n", v, f_name]);
                #fmt["\ns.emit_rec_field(\"%s\", %uu, {||%s})\n",
                     f_name, i, efld]
            };
            #fmt["\ns.emit_rec({||%s})\n", self.blk_expr(stmts)]
          }
          ty::ty_tup(tys) {
            let (pat, stmts) = self.serialize_arm("", "emit_tup_elt", tys);
            #fmt["s.emit_tup(%uu, {|| alt %s { \
                    %s %s \
                  }})", vec::len(tys), v, pat, self.blk_expr(stmts)]
          }
          ty::ty_constr(t, _) {
            self.serialize_ty(t, v)
          }
          ty::ty_ptr(_) |
          ty::ty_rptr(_,_) |
          ty::ty_fn(_) |
          ty::ty_iface(_, _) |
          ty::ty_res(_, _, _) |
          ty::ty_var(_) | ty::ty_param(_, _) |
          ty::ty_self(_) | ty::ty_type | ty::ty_send_type |
          ty::ty_opaque_closure_ptr(_) | ty::ty_opaque_box {
            fail #fmt["Unhandled type %s", ty0_str]
          }
        };

        let item = #fmt["/*%s*/ fn %s\n\
                         <S:std::serialization::serializer>\n\
                            (s: S,\n\
                            v: %s) {\n\
                             %s;\n\
                         }", ty0_str, name, ty0_str, body_node];
        self.add_item(item);
    }

    fn serialize_enum(v: ast_expr,
                      id: ast::def_id,
                      tps: [ty::t]) -> ast_expr {
        let variants = ty::substd_enum_variants(self.tcx, id, tps);

        let idx = 0u;
        let arms = vec::map(variants) {|variant|
            let item_path = ty::item_path(self.tcx, variant.id);
            let v_path = ast_map::path_to_str(item_path);
            let n_args = vec::len(variant.args);
            let (v_pat, stmts) = {
                if n_args == 0u {
                    (v_path, [])
                } else {
                    self.serialize_arm(v_path, "emit_enum_variant_arg",
                                       variant.args)
                }
            };

            let v_id = idx;
            idx += 1u;

            #fmt["%s {\n\
                    s.emit_enum_variant(\"%s\", %uu, %uu, {||\n\
                      %s\n\
                    })\n\
                  }", v_pat, v_path, v_id, n_args, self.blk(stmts)]
        };

        let enum_name = ast_map::path_to_str(ty::item_path(self.tcx, id));
        #fmt["\ns.emit_enum(\"%s\", {||\n\
                alt %s {\n\
                  %s\n\
                }\n\
              })\n", enum_name, v, str::connect(arms, "\n")]
    }

    fn serialize_arm(v_path: str, emit_fn: str, args: [ty::t])
        -> (ast_pat, [ast_stmt]) {
        let n_args = vec::len(args);
        let arg_nms = vec::from_fn(n_args) {|i| #fmt["v%u", i] };
        let v_pat =
            #fmt["\n%s(%s)\n", v_path, str::connect(arg_nms, ",")];
        let stmts = vec::from_fn(n_args) {|i|
            let arg_ty = args[i];
            let serialize_expr =
                self.serialize_ty(arg_ty, arg_nms[i]);
            #fmt["\ns.%s(%uu, {||\n%s\n})\n", emit_fn, i, serialize_expr]
        };
        (v_pat, stmts)
    }
}

impl deserialize_methods for serialize_ctx {
    fn mk_deserialize_named_item_fn(name: str) -> str {
        self.exec_named_item_fn(name) {|item_ty|
            let fname = self.mk_deserialize_ty_fn(item_ty);

            let ty_str = ppaux::ty_to_str(self.tcx, item_ty);
            check str::is_not_empty("::");
            let namep = str::replace(name, "::", "_");
            let item = #fmt["fn deserialize_%s\
                                 <S:std::serialization::deserializer>\n\
                                 (s: S) -> %s {\n\
                                   %s(s)\
                                 }", namep, ty_str, fname];
            self.add_item(item);

            fname
        }
    }

    // Generates a function to serialize the given type.
    // Returns an AST fragment that names this function.
    fn deserialize_ty(ty0: ty::t) -> ast_expr {
        let fname = self.mk_deserialize_ty_fn(ty0);
        let ty0_str = ppaux::ty_to_str(self.tcx, ty0);
        #fmt["\n/*%s*/ %s(s)\n", ty0_str, fname]
    }

    fn mk_deserialize_ty_fn(ty0: ty::t) -> str {
        self.memoize(self.deserialize_tyfns, "deserialize", ty0) {|name|
            self.mk_deserialize_ty_fn0(ty0, name)
        }
    }

    fn mk_deserialize_ty_fn0(ty0: ty::t, name: str) {
        let ty0_str = ppaux::ty_to_str(self.tcx, ty0);
        let body_node = alt ty::get(ty0).struct {
          ty::ty_nil | ty::ty_bot { "()" }

          ty::ty_int(ast::ty_i)   { #fmt["s.read_int()"] }
          ty::ty_int(ast::ty_i64) { #fmt["s.read_i64()"] }
          ty::ty_int(ast::ty_i32) { #fmt["s.read_i32()"] }
          ty::ty_int(ast::ty_i16) { #fmt["s.read_i16()"] }
          ty::ty_int(ast::ty_i8)  { #fmt["s.read_i8()"]  }

          ty::ty_int(ast::ty_char) { #fmt["s.read_char()"] }

          ty::ty_uint(ast::ty_u)   { #fmt["s.read_uint()"] }
          ty::ty_uint(ast::ty_u64) { #fmt["s.read_u64()"] }
          ty::ty_uint(ast::ty_u32) { #fmt["s.read_u32()"] }
          ty::ty_uint(ast::ty_u16) { #fmt["s.read_u16()"] }
          ty::ty_uint(ast::ty_u8)  { #fmt["s.read_u8()"]  }

          ty::ty_float(ast::ty_f64) { #fmt["s.read_f64()"] }
          ty::ty_float(ast::ty_f32) { #fmt["s.read_f32()"] }
          ty::ty_float(ast::ty_f)   { #fmt["s.read_float()"] }

          ty::ty_bool { #fmt["s.read_bool()"] }

          ty::ty_str { #fmt["s.read_str()"] }

          ty::ty_enum(def_id, tps) { self.deserialize_enum(def_id, tps) }
          ty::ty_box(mt) {
            let s = self.deserialize_ty(mt.ty);
            #fmt["\ns.read_box({||@%s})\n", s]
          }
          ty::ty_uniq(mt) {
            let s = self.deserialize_ty(mt.ty);
            #fmt["\ns.read_uniq({||~%s})\n", s]
          }
          ty::ty_vec(mt) {
            let selem = self.deserialize_ty(mt.ty);
            #fmt["s.read_vec({|len|\n\
                    vec::from_fn(len, {|i|\n\
                      s.read_vec_elt(i, {||\n\
                        %s\n\
                  })})})", selem]
          }
          ty::ty_class(_, _) {
            fail "TODO--implement class";
          }
          ty::ty_rec(fields) {
            let i = 0u;
            let flds = vec::map(fields) {|field|
                let f_name = field.ident;
                let f_ty = field.mt.ty;
                let rfld = self.deserialize_ty(f_ty);
                let idx = i;
                i += 1u;
                #fmt["\n%s: s.read_rec_field(\"%s\", %uu, {||\n%s\n})\n",
                     f_name, f_name, idx, rfld]
            };
            #fmt["\ns.read_rec({||{\n%s\n}})\n", str::connect(flds, ",")]
          }
          ty::ty_tup(tys) {
            let rexpr = self.deserialize_arm("", "read_tup_elt", tys);
            #fmt["\ns.read_tup(%uu, {||\n%s\n})\n", vec::len(tys), rexpr]
          }
          ty::ty_constr(t, _) {
            self.deserialize_ty(t)
          }
          ty::ty_ptr(_) |
          ty::ty_rptr(_,_) |
          ty::ty_fn(_) |
          ty::ty_iface(_, _) |
          ty::ty_res(_, _, _) |
          ty::ty_var(_) | ty::ty_param(_, _) |
          ty::ty_self(_) | ty::ty_type | ty::ty_send_type |
          ty::ty_opaque_closure_ptr(_) | ty::ty_opaque_box {
            fail #fmt["Unhandled type %s", ty0_str]
          }
        };

        let item = #fmt["/*%s*/\n\
                         fn %s\n\
                         <S:std::serialization::deserializer>(s: S)\n\
                         -> %s {\n\
                             %s\n\
                         }", ty0_str, name, ty0_str, body_node];
        self.add_item(item);
    }

    fn deserialize_enum(id: ast::def_id,
                        tps: [ty::t]) -> ast_expr {
        let variants = ty::substd_enum_variants(self.tcx, id, tps);

        let arms = vec::from_fn(vec::len(variants)) {|v_id|
            let variant = variants[v_id];
            let item_path = ty::item_path(self.tcx, variant.id);
            let v_path = ast_map::path_to_str(item_path);
            let n_args = vec::len(variant.args);
            let rexpr = {
                if n_args == 0u {
                    #fmt["\n%s\n", v_path]
                } else {
                    self.deserialize_arm(v_path, "read_enum_variant_arg",
                                         variant.args)
                }
            };

            #fmt["\n%uu { %s }\n", v_id, rexpr]
        };

        let enum_name = ast_map::path_to_str(ty::item_path(self.tcx, id));
        #fmt["s.read_enum(\"%s\", {||\n\
                s.read_enum_variant({|v_id|\n\
                  alt check v_id {\n\
                    %s\n\
                  }\n\
                })})", enum_name, str::connect(arms, "\n")]
    }

    fn deserialize_arm(v_path: str, read_fn: str, args: [ty::t])
        -> ast_expr {
        let exprs = vec::from_fn(vec::len(args)) {|i|
            let rexpr = self.deserialize_ty(args[i]);
            #fmt["\ns.%s(%uu, {||%s})\n", read_fn, i, rexpr]
        };
        #fmt["\n%s(%s)\n", v_path, str::connect(exprs, ",")]
    }
}

fn main(argv: [str]) {
    let {crate, tcx, roots} = parse(argv);
    let sctx: serialize_ctx = {
        // let cm = codemap::new_codemap();
        // let handler = diagnostic::mk_handler(option::none);
        // let psess: parser::parse_sess = @{
        //     cm: cm,
        //     mutable next_id: 1,
        //     span_diagnostic: diagnostic::mk_span_handler(handler, cm),
        //     mutable chpos: 0u,
        //     mutable byte_pos: 0u
        // };
        {crate: crate,
         tcx: tcx,
         serialize_tyfns: ty::new_ty_hash::<str>(),
         deserialize_tyfns: ty::new_ty_hash::<str>(),
         mutable item_fns: [],
         mutable constants: []}
    };

    vec::iter(roots) {|root|
        sctx.mk_serialize_named_item_fn(root);
        sctx.mk_deserialize_named_item_fn(root);
    }

    let stdout = io::stdout();
    vec::iter(copy sctx.item_fns) {|item|
        stdout.write_str(#fmt["%s\n", item])
    }
}
