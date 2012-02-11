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
import std::io;
import std::io::writer_util;
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

type tp_map = map<ast::node_id, ty::t>;

type serialize_ctx = {
    crate: @ast::crate,
    tcx: ty::ctxt,

    tyfns: hashmap<ty::t, str>,
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
    // fn session() -> parser::parse_sess { self.psess }

    fn add_item(item: ast_item) {
        self.item_fns += [item];
    }

    fn mk_serialize_named_item_fn(name: str) -> str {
        let names = str::split_str(name, "::");
        let item = lookup(self.crate.node.module, 0u, names);
        let def_id = {crate: ast::local_crate, node: item.id};
        self.mk_serialize_item_fn(def_id, [])
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

    fn mk_serialize_item_fn(id: ast::def_id,
                            tps: [ty::t]) -> str {
        let item_ty = self.instantiate(id, tps);
        self.mk_serialize_ty_fn(item_ty)
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
        #fmt["%s(cx, %s)", fname, v]
    }

    fn mk_serialize_ty_fn(ty0: ty::t) -> str {
        // check for existing function
        alt self.tyfns.find(ty0) {
          some(name) { ret name; }
          none { /* fallthrough */ }
        }

        // define the name and insert into the hashtable
        // in case of recursive calls:
        let id = self.tyfns.size();
        let ty0_str = ppaux::ty_to_str(self.tcx, ty0);
        #debug["ty0_str = %s / ty0 = %?", ty0_str, ty0];
        let name = #fmt["serialize_%u /*%s*/", id, ty0_str];
        self.tyfns.insert(ty0, name);
        let v = "v";

        let body_node = alt ty::get(ty0).struct {
          ty::ty_nil | ty::ty_bot { "()" }
          ty::ty_int(_) { #fmt["s.emit_i64(%s as i64)", v] }
          ty::ty_uint(_) { #fmt["s.emit_u64(%s as u64)", v] }
          ty::ty_float(_) { #fmt["s.emit_f64(%s as f64)", v] }
          ty::ty_bool { #fmt["s.emit_bool(%s)", v] }
          ty::ty_str { #fmt["s.emit_str(%s)", v] }
          ty::ty_enum(def_id, tps) { self.serialize_enum(v, def_id, tps) }
          ty::ty_box(mt) {
            let s = self.serialize_ty(mt.ty, #fmt["*%s", v]);
            #fmt["s.emit_box({||%s})", s]
          }
          ty::ty_uniq(mt) {
            let s = self.serialize_ty(mt.ty, #fmt["*%s", v]);
            #fmt["s.emit_uniq({||%s})", s]
          }
          ty::ty_vec(mt) {
            let selem = self.serialize_ty(mt.ty, "i");
            #fmt["s.emit_vec(vec::len(v), {|| \
                  uint::range(0, vec::len(v), {|i| \
                  s.emit_vec_elt(i, {||\
                  %s;\
                  })})})", selem]
          }
          ty::ty_class(_, _) {
            fail "TODO--implement class";
          }
          ty::ty_rec(fields) {
            let stmts = vec::map(fields) {|field|
                let f_name = field.ident;
                let f_ty = field.mt.ty;
                self.serialize_ty(f_ty, #fmt["%s.%s", v, f_name])
            };
            #fmt["s.emit_rec({||%s})", self.blk_expr(stmts)]
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
          ty::ty_fn(_) |
          ty::ty_iface(_, _) |
          ty::ty_res(_, _, _) |
          ty::ty_var(_) | ty::ty_param(_, _) |
          ty::ty_self(_) | ty::ty_type | ty::ty_send_type |
          ty::ty_opaque_closure_ptr(_) | ty::ty_opaque_box {
            fail #fmt["Unhandled type %s", ty0_str]
          }
        };

        let item = #fmt["fn %s<S:std::serialization::serializer>\
                            (s: S, v: %s) {\
                             %s;\
                         }", name, ty0_str, body_node];
        self.add_item(item);
        ret name;
    }

    fn serialize_enum(v: ast_expr,
                      id: ast::def_id,
                      tps: [ty::t]) -> ast_expr {
        let variants = ty::substd_enum_variants(self.tcx, id, tps);

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

            let v_ident = ast_map::path_to_str_with_sep(item_path, "_");
            let v_const = #fmt["at_%s", v_ident];

            #fmt["%s { \
                    start_variant(cx, %s); \
                    %s \
                    end_variant(cx, %s); \
                  }", v_pat, v_const, self.blk(stmts), v_const]
        };

        #fmt["alt %s { \
                %s \
              }", v, str::connect(arms, "\n")]
    }

    fn serialize_arm(v_path: str, emit_fn: str, args: [ty::t])
        -> (ast_pat, [ast_stmt]) {
        let n_args = vec::len(args);
        let arg_nms = vec::init_fn(n_args) {|i| #fmt["v%u", i] };
        let v_pat =
            #fmt["%s(%s)", v_path, str::connect(arg_nms, ", ")];
        let stmts = vec::init_fn(n_args) {|i|
            let arg_ty = args[i];
            let serialize_expr =
                self.serialize_ty(arg_ty, arg_nms[i]);
            #fmt["s.%s(%uu, {|| %s })", emit_fn, i, serialize_expr]
        };
        (v_pat, stmts)
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
         tyfns: ty::new_ty_hash::<str>(),
         mutable item_fns: [],
         mutable constants: []}
    };

    vec::iter(roots) {|root|
        sctx.mk_serialize_named_item_fn(root);
    }

    let stdout = io::stdout();
    vec::iter(copy sctx.item_fns) {|item|
        stdout.write_str(#fmt["%s\n", item])
    }
}