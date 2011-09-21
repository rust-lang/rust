
import std::{vec, int, io, str, uint, option};
import parse::lexer;
import syntax::codemap::codemap;
import syntax::visit;
import ast;
import ast_util;
import option::{some, none};
import pp::{printer, break_offset, word, huge_word, zero_word,
            space, zerobreak, hardbreak, breaks, consistent,
            inconsistent, eof};

// The ps is stored here to prevent recursive type.
// FIXME use a nominal tag instead
tag ann_node {
    node_block(ps, ast::blk);
    node_item(ps, @ast::item);
    node_expr(ps, @ast::expr);
    node_pat(ps, @ast::pat);
}
type pp_ann = {pre: fn(ann_node), post: fn(ann_node)};

fn no_ann() -> pp_ann {
    fn ignore(_node: ann_node) { }
    ret {pre: ignore, post: ignore};
}

type ps =
    @{s: pp::printer,
      cm: option::t<codemap>,
      comments: option::t<[lexer::cmnt]>,
      literals: option::t<[lexer::lit]>,
      mutable cur_cmnt: uint,
      mutable cur_lit: uint,
      mutable boxes: [pp::breaks],
      ann: pp_ann};

fn ibox(s: ps, u: uint) { s.boxes += [pp::inconsistent]; pp::ibox(s.s, u); }

fn end(s: ps) { vec::pop(s.boxes); pp::end(s.s); }

fn rust_printer(writer: io::writer) -> ps {
    let boxes: [pp::breaks] = [];
    ret @{s: pp::mk_printer(writer, default_columns),
          cm: none::<codemap>,
          comments: none::<[lexer::cmnt]>,
          literals: none::<[lexer::lit]>,
          mutable cur_cmnt: 0u,
          mutable cur_lit: 0u,
          mutable boxes: boxes,
          ann: no_ann()};
}

const indent_unit: uint = 4u;
const alt_indent_unit: uint = 2u;

const default_columns: uint = 78u;

// Requires you to pass an input filename and reader so that
// it can scan the input text for comments and literals to
// copy forward.
fn print_crate(cm: codemap, crate: @ast::crate, filename: str, in: io::reader,
               out: io::writer, ann: pp_ann) {
    let boxes: [pp::breaks] = [];
    let r = lexer::gather_comments_and_literals(cm, filename, in);
    let s =
        @{s: pp::mk_printer(out, default_columns),
          cm: some(cm),
          comments: some(r.cmnts),
          literals: some(r.lits),
          mutable cur_cmnt: 0u,
          mutable cur_lit: 0u,
          mutable boxes: boxes,
          ann: ann};
    print_mod(s, crate.node.module, crate.node.attrs);
    print_remaining_comments(s);
    eof(s.s);
}

fn ty_to_str(ty: @ast::ty) -> str { be to_str(ty, print_type); }

fn pat_to_str(pat: @ast::pat) -> str { be to_str(pat, print_pat); }

fn expr_to_str(e: @ast::expr) -> str { be to_str(e, print_expr); }

fn stmt_to_str(s: ast::stmt) -> str { be to_str(s, print_stmt); }

fn item_to_str(i: @ast::item) -> str { be to_str(i, print_item); }

fn path_to_str(p: ast::path) -> str {
    be to_str(p, bind print_path(_, _, false));
}

fn fun_to_str(f: ast::_fn, name: ast::ident, params: [ast::ty_param]) -> str {
    let writer = io::string_writer();
    let s = rust_printer(writer.get_writer());
    print_fn(s, f.decl, f.proto, name, params, f.decl.constraints);
    eof(s.s);
    ret writer.get_str();
}

fn block_to_str(blk: ast::blk) -> str {
    let writer = io::string_writer();
    let s = rust_printer(writer.get_writer());
    // containing cbox, will be closed by print-block at }

    cbox(s, indent_unit);
    // head-ibox, will be closed by print-block after {

    ibox(s, 0u);
    print_block(s, blk);
    eof(s.s);
    ret writer.get_str();
}

fn meta_item_to_str(mi: ast::meta_item) -> str {
    ret to_str(@mi, print_meta_item);
}

fn attribute_to_str(attr: ast::attribute) -> str {
    be to_str(attr, print_attribute);
}

fn cbox(s: ps, u: uint) { s.boxes += [pp::consistent]; pp::cbox(s.s, u); }

fn box(s: ps, u: uint, b: pp::breaks) { s.boxes += [b]; pp::box(s.s, u, b); }

fn nbsp(s: ps) { word(s.s, " "); }

fn word_nbsp(s: ps, w: str) { word(s.s, w); nbsp(s); }

fn word_space(s: ps, w: str) { word(s.s, w); space(s.s); }

fn popen(s: ps) { word(s.s, "("); }

fn pclose(s: ps) { word(s.s, ")"); }

fn head(s: ps, w: str) {
    // outer-box is consistent
    cbox(s, indent_unit);
    // head-box is inconsistent
    ibox(s, str::char_len(w) + 1u);
    // keyword that starts the head
    word_nbsp(s, w);
}

fn bopen(s: ps) {
    word(s.s, "{");
    end(s); // close the head-box
}

fn bclose_(s: ps, span: codemap::span, indented: uint) {
    maybe_print_comment(s, span.hi);
    break_offset_if_not_bol(s, 1u, -(indented as int));
    word(s.s, "}");
    end(s); // close the outer-box
}
fn bclose(s: ps, span: codemap::span) { bclose_(s, span, indent_unit); }

fn is_begin(s: ps) -> bool {
    alt s.s.last_token() { pp::BEGIN(_) { true } _ { false } }
}

fn is_end(s: ps) -> bool {
    alt s.s.last_token() { pp::END. { true } _ { false } }
}

fn is_bol(s: ps) -> bool {
    ret s.s.last_token() == pp::EOF ||
            s.s.last_token() == pp::hardbreak_tok();
}

fn hardbreak_if_not_bol(s: ps) { if !is_bol(s) { hardbreak(s.s); } }
fn space_if_not_bol(s: ps) { if !is_bol(s) { space(s.s); } }
fn break_offset_if_not_bol(s: ps, n: uint, off: int) {
    if !is_bol(s) {
        break_offset(s.s, n, off);
    } else {
        if off != 0 && s.s.last_token() == pp::hardbreak_tok() {
            // We do something pretty sketchy here: tuck the nonzero
            // offset-adjustment we were going to deposit along with the
            // break into the previous hardbreak.
            s.s.replace_last_token(pp::hardbreak_tok_offset(off));
        }
    }
}

// Synthesizes a comment that was not textually present in the original source
// file.
fn synth_comment(s: ps, text: str) {
    word(s.s, "/*");
    space(s.s);
    word(s.s, text);
    space(s.s);
    word(s.s, "*/");
}

fn commasep<IN>(s: ps, b: breaks, elts: [IN], op: fn(ps, IN)) {
    box(s, 0u, b);
    let first = true;
    for elt: IN in elts {
        if first { first = false; } else { word_space(s, ","); }
        op(s, elt);
    }
    end(s);
}


fn commasep_cmnt<IN>(s: ps, b: breaks, elts: [IN], op: fn(ps, IN),
                     get_span: fn(IN) -> codemap::span) {
    box(s, 0u, b);
    let len = vec::len::<IN>(elts);
    let i = 0u;
    for elt: IN in elts {
        maybe_print_comment(s, get_span(elt).hi);
        op(s, elt);
        i += 1u;
        if i < len {
            word(s.s, ",");
            maybe_print_trailing_comment(s, get_span(elt),
                                         some(get_span(elts[i]).hi));
            space_if_not_bol(s);
        }
    }
    end(s);
}

fn commasep_exprs(s: ps, b: breaks, exprs: [@ast::expr]) {
    fn expr_span(expr: @ast::expr) -> codemap::span { ret expr.span; }
    commasep_cmnt(s, b, exprs, print_expr, expr_span);
}

fn print_mod(s: ps, _mod: ast::_mod, attrs: [ast::attribute]) {
    print_inner_attributes(s, attrs);
    for vitem: @ast::view_item in _mod.view_items {
        print_view_item(s, vitem);
    }
    for item: @ast::item in _mod.items { print_item(s, item); }
}

fn print_native_mod(s: ps, nmod: ast::native_mod, attrs: [ast::attribute]) {
    print_inner_attributes(s, attrs);
    for vitem: @ast::view_item in nmod.view_items {
        print_view_item(s, vitem);
    }
    for item: @ast::native_item in nmod.items { print_native_item(s, item); }
}

fn print_type(s: ps, ty: @ast::ty) {
    maybe_print_comment(s, ty.span.lo);
    ibox(s, 0u);
    alt ty.node {
      ast::ty_nil. { word(s.s, "()"); }
      ast::ty_bool. { word(s.s, "bool"); }
      ast::ty_bot. { word(s.s, "!"); }
      ast::ty_int. { word(s.s, "int"); }
      ast::ty_uint. { word(s.s, "uint"); }
      ast::ty_float. { word(s.s, "float"); }
      ast::ty_machine(tm) { word(s.s, ast_util::ty_mach_to_str(tm)); }
      ast::ty_char. { word(s.s, "char"); }
      ast::ty_str. { word(s.s, "str"); }
      ast::ty_box(mt) { word(s.s, "@"); print_mt(s, mt); }
      ast::ty_uniq(mt) { word(s.s, "~"); print_mt(s, mt); }
      ast::ty_vec(mt) {
        word(s.s, "[");
        alt mt.mut {
          ast::mut. { word_space(s, "mutable"); }
          ast::maybe_mut. { word_space(s, "mutable?"); }
          ast::imm. { }
        }
        print_type(s, mt.ty);
        word(s.s, "]");
      }
      ast::ty_ptr(mt) { word(s.s, "*"); print_mt(s, mt); }
      ast::ty_task. { word(s.s, "task"); }
      ast::ty_port(t) {
        word(s.s, "port<");
        print_type(s, t);
        word(s.s, ">");
      }
      ast::ty_chan(t) {
        word(s.s, "chan<");
        print_type(s, t);
        word(s.s, ">");
      }
      ast::ty_rec(fields) {
        word(s.s, "{");
        fn print_field(s: ps, f: ast::ty_field) {
            cbox(s, indent_unit);
            print_mutability(s, f.node.mt.mut);
            word(s.s, f.node.ident);
            word_space(s, ":");
            print_type(s, f.node.mt.ty);
            end(s);
        }
        fn get_span(f: ast::ty_field) -> codemap::span { ret f.span; }
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        word(s.s, "}");
      }
      ast::ty_tup(elts) {
        popen(s);
        commasep(s, inconsistent, elts, print_type);
        pclose(s);
      }
      ast::ty_fn(proto, inputs, output, cf, constrs) {
        print_ty_fn(s, proto, none::<str>, inputs, output, cf, constrs);
      }
      ast::ty_obj(methods) {
        head(s, "obj");
        bopen(s);
        for m: ast::ty_method in methods {
            hardbreak_if_not_bol(s);
            cbox(s, indent_unit);
            maybe_print_comment(s, m.span.lo);
            print_ty_fn(s, m.node.proto, some(m.node.ident), m.node.inputs,
                        m.node.output, m.node.cf, m.node.constrs);
            word(s.s, ";");
            end(s);
        }
        bclose(s, ty.span);
      }
      ast::ty_path(path, _) { print_path(s, path, false); }
      ast::ty_type. { word(s.s, "type"); }
      ast::ty_constr(t, cs) {
        print_type(s, t);
        space(s.s);
        word(s.s, ast_ty_constrs_str(cs));
      }
    }
    end(s);
}

fn print_native_item(s: ps, item: @ast::native_item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    alt item.node {
      ast::native_item_ty. {
        ibox(s, indent_unit);
        ibox(s, 0u);
        word_nbsp(s, "type");
        word(s.s, item.ident);
        end(s); // end the inner ibox
        word(s.s, ";");
        end(s); // end the outer ibox

      }






      ast::native_item_fn(lname, decl, typarams) {
        print_fn(s, decl, ast::proto_fn, item.ident, typarams,
                 decl.constraints);
        alt lname {
          none. { }
          some(ss) { space(s.s); word_space(s, "="); print_string(s, ss); }
        }
        end(s); // end head-ibox
        word(s.s, ";");
        end(s); // end the outer fn box
      }
    }
}

fn print_item(s: ps, item: @ast::item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    let ann_node = node_item(s, item);
    s.ann.pre(ann_node);
    alt item.node {
      ast::item_const(ty, expr) {
        head(s, "const");
        word_space(s, item.ident + ":");
        print_type(s, ty);
        space(s.s);
        end(s); // end the head-ibox

        word_space(s, "=");
        print_expr(s, expr);
        word(s.s, ";");
        end(s); // end the outer cbox

      }
      ast::item_fn(_fn, typarams) {
        print_fn(s, _fn.decl, _fn.proto, item.ident, typarams,
                 _fn.decl.constraints);
        word(s.s, " ");
        print_block(s, _fn.body);
      }
      ast::item_mod(_mod) {
        head(s, "mod");
        word_nbsp(s, item.ident);
        bopen(s);
        print_mod(s, _mod, item.attrs);
        bclose(s, item.span);
      }
      ast::item_native_mod(nmod) {
        head(s, "native");
        alt nmod.abi {
          ast::native_abi_llvm. { word_nbsp(s, "\"llvm\""); }
          ast::native_abi_rust. { word_nbsp(s, "\"rust\""); }
          ast::native_abi_cdecl. { word_nbsp(s, "\"cdecl\""); }
          ast::native_abi_rust_intrinsic. {
            word_nbsp(s, "\"rust-intrinsic\"");
          }
          ast::native_abi_x86stdcall. { word_nbsp(s, "\"x86stdcall\""); }
        }
        word_nbsp(s, "mod");
        word_nbsp(s, item.ident);
        if !str::eq(nmod.native_name, item.ident) {
            word_space(s, "=");
            print_string(s, nmod.native_name);
            nbsp(s);
        }
        bopen(s);
        print_native_mod(s, nmod, item.attrs);
        bclose(s, item.span);
      }
      ast::item_ty(ty, params) {
        ibox(s, indent_unit);
        ibox(s, 0u);
        word_nbsp(s, "type");
        word(s.s, item.ident);
        print_type_params(s, params);
        end(s); // end the inner ibox

        space(s.s);
        word_space(s, "=");
        print_type(s, ty);
        word(s.s, ";");
        end(s); // end the outer ibox
      }
      ast::item_tag(variants, params) {
        let newtype =
            vec::len(variants) == 1u &&
                str::eq(item.ident, variants[0].node.name) &&
                vec::len(variants[0].node.args) == 1u;
        if newtype {
            ibox(s, indent_unit);
            word_space(s, "tag");
        } else { head(s, "tag"); }
        word(s.s, item.ident);
        print_type_params(s, params);
        space(s.s);
        if newtype {
            word_space(s, "=");
            print_type(s, variants[0].node.args[0].ty);
            word(s.s, ";");
            end(s);
        } else {
            bopen(s);
            for v: ast::variant in variants {
                space_if_not_bol(s);
                maybe_print_comment(s, v.span.lo);
                word(s.s, v.node.name);
                if vec::len(v.node.args) > 0u {
                    popen(s);
                    fn print_variant_arg(s: ps, arg: ast::variant_arg) {
                        print_type(s, arg.ty);
                    }
                    commasep(s, consistent, v.node.args, print_variant_arg);
                    pclose(s);
                }
                word(s.s, ";");
                maybe_print_trailing_comment(s, v.span, none::<uint>);
            }
            bclose(s, item.span);
        }
      }
      ast::item_obj(_obj, params, _) {
        head(s, "obj");
        word(s.s, item.ident);
        print_type_params(s, params);
        popen(s);
        fn print_field(s: ps, field: ast::obj_field) {
            ibox(s, indent_unit);
            print_mutability(s, field.mut);
            word_space(s, field.ident + ":");
            print_type(s, field.ty);
            end(s);
        }
        fn get_span(f: ast::obj_field) -> codemap::span { ret f.ty.span; }
        commasep_cmnt(s, consistent, _obj.fields, print_field, get_span);
        pclose(s);
        space(s.s);
        bopen(s);
        for meth: @ast::method in _obj.methods {
            let typarams: [ast::ty_param] = [];
            hardbreak_if_not_bol(s);
            maybe_print_comment(s, meth.span.lo);
            print_fn(s, meth.node.meth.decl, meth.node.meth.proto,
                     meth.node.ident, typarams, []);
            word(s.s, " ");
            print_block(s, meth.node.meth.body);
        }
        bclose(s, item.span);
      }
      ast::item_res(dt, dt_id, tps, ct_id) {
        head(s, "resource");
        word(s.s, item.ident);
        print_type_params(s, tps);
        popen(s);
        word_space(s, dt.decl.inputs[0].ident + ":");
        print_type(s, dt.decl.inputs[0].ty);
        pclose(s);
        space(s.s);
        print_block(s, dt.body);
      }
    }
    s.ann.post(ann_node);
}

fn print_outer_attributes(s: ps, attrs: [ast::attribute]) {
    let count = 0;
    for attr: ast::attribute in attrs {
        alt attr.node.style {
          ast::attr_outer. { print_attribute(s, attr); count += 1; }
          _ {/* fallthrough */ }
        }
    }
    if count > 0 { hardbreak_if_not_bol(s); }
}

fn print_inner_attributes(s: ps, attrs: [ast::attribute]) {
    let count = 0;
    for attr: ast::attribute in attrs {
        alt attr.node.style {
          ast::attr_inner. {
            print_attribute(s, attr);
            word(s.s, ";");
            count += 1;
          }
          _ {/* fallthrough */ }
        }
    }
    if count > 0 { hardbreak_if_not_bol(s); }
}

fn print_attribute(s: ps, attr: ast::attribute) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, attr.span.lo);
    word(s.s, "#[");
    print_meta_item(s, @attr.node.value);
    word(s.s, "]");
}

fn print_stmt(s: ps, st: ast::stmt) {
    maybe_print_comment(s, st.span.lo);
    alt st.node {
      ast::stmt_decl(decl, _) { print_decl(s, decl); }
      ast::stmt_expr(expr, _) { space_if_not_bol(s); print_expr(s, expr); }
    }
    if parse::parser::stmt_ends_with_semi(st) { word(s.s, ";"); }
    maybe_print_trailing_comment(s, st.span, none::<uint>);
}

fn print_block(s: ps, blk: ast::blk) {
    print_possibly_embedded_block(s, blk, block_normal, indent_unit);
}

tag embed_type { block_macro; block_block_fn; block_normal; }

fn print_possibly_embedded_block(s: ps, blk: ast::blk, embedded: embed_type,
                                 indented: uint) {
    alt blk.node.rules { ast::unchecked. { word(s.s, "unchecked"); } _ { } }

    maybe_print_comment(s, blk.span.lo);
    let ann_node = node_block(s, blk);
    s.ann.pre(ann_node);
    alt embedded {
      block_macro. { word(s.s, "#{"); end(s); }
      block_block_fn. { end(s); }
      block_normal. { bopen(s); }
    }

    for st: @ast::stmt in blk.node.stmts {
        print_stmt(s, *st);
    }
    alt blk.node.expr {
      some(expr) {
        space_if_not_bol(s);
        print_expr(s, expr);
        maybe_print_trailing_comment(s, expr.span, some(blk.span.hi));
      }
      _ { }
    }
    bclose_(s, blk.span, indented);
    s.ann.post(ann_node);
}

// ret and fail, without arguments cannot appear is the discriminant of if,
// alt, do, & while unambiguously without being parenthesized
fn print_maybe_parens_discrim(s: ps, e: @ast::expr) {
    let disambig = alt e.node {
      ast::expr_ret(none.) | ast::expr_put(none.) |
      ast::expr_fail(none.) { true }
      _ { false }
    };
    if disambig { popen(s); }
    print_expr(s, e);
    if disambig { pclose(s); }
}

fn print_if(s: ps, test: @ast::expr, blk: ast::blk,
            elseopt: option::t<@ast::expr>, chk: bool) {
    head(s, "if");
    if chk { word_nbsp(s, "check"); }
    print_maybe_parens_discrim(s, test);
    space(s.s);
    print_block(s, blk);
    fn do_else(s: ps, els: option::t<@ast::expr>) {
        alt els {
          some(_else) {
            alt _else.node {





              // "another else-if"
              ast::expr_if(i, t, e) {
                cbox(s, indent_unit - 1u);
                ibox(s, 0u);
                word(s.s, " else if ");
                print_maybe_parens_discrim(s, i);
                space(s.s);
                print_block(s, t);
                do_else(s, e);
              }





              // "final else"
              ast::expr_block(b) {
                cbox(s, indent_unit - 1u);
                ibox(s, 0u);
                word(s.s, " else ");
                print_block(s, b);
              }
            }
          }
          _ {/* fall through */ }
        }
    }
    do_else(s, elseopt);
}

fn print_mac(s: ps, m: ast::mac) {
    alt m.node {
      ast::mac_invoc(path, arg, body) {
        word(s.s, "#");
        print_path(s, path, false);
        alt arg.node { ast::expr_vec(_, _) { } _ { word(s.s, " "); } }
        print_expr(s, arg);
        // FIXME: extension 'body'
      }
      ast::mac_embed_type(ty) {
        word(s.s, "#<");
        print_type(s, ty);
        word(s.s, ">");
      }
      ast::mac_embed_block(blk) {
        print_possibly_embedded_block(s, blk, block_normal, indent_unit);
      }
      ast::mac_ellipsis. { word(s.s, "..."); }
    }
}

fn print_expr(s: ps, expr: @ast::expr) {
    maybe_print_comment(s, expr.span.lo);
    ibox(s, indent_unit);
    let ann_node = node_expr(s, expr);
    s.ann.pre(ann_node);
    alt expr.node {
      ast::expr_vec(exprs, mut) {
        ibox(s, indent_unit);
        word(s.s, "[");
        if mut == ast::mut {
            word(s.s, "mutable");
            if vec::len(exprs) > 0u { nbsp(s); }
        }
        commasep_exprs(s, inconsistent, exprs);
        word(s.s, "]");
        end(s);
      }
      ast::expr_rec(fields, wth) {
        fn print_field(s: ps, field: ast::field) {
            ibox(s, indent_unit);
            if field.node.mut == ast::mut { word_nbsp(s, "mutable"); }
            word(s.s, field.node.ident);
            word_space(s, ":");
            print_expr(s, field.node.expr);
            end(s);
        }
        fn get_span(field: ast::field) -> codemap::span { ret field.span; }
        word(s.s, "{");
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        alt wth {
          some(expr) {
            if vec::len(fields) > 0u { space(s.s); }
            ibox(s, indent_unit);
            word_space(s, "with");
            print_expr(s, expr);
            end(s);
          }
          _ { }
        }
        word(s.s, "}");
      }
      ast::expr_tup(exprs) {
        popen(s);
        commasep_exprs(s, inconsistent, exprs);
        pclose(s);
      }
      ast::expr_call(func, args) {
        print_expr_parens_if_not_bot(s, func);
        popen(s);
        commasep_exprs(s, inconsistent, args);
        pclose(s);
      }
      ast::expr_self_method(ident) {
        word(s.s, "self.");
        print_ident(s, ident);
      }
      ast::expr_bind(func, args) {
        fn print_opt(s: ps, expr: option::t<@ast::expr>) {
            alt expr {
              some(expr) { print_expr(s, expr); }
              _ { word(s.s, "_"); }
            }
        }
        word_nbsp(s, "bind");
        print_expr(s, func);
        popen(s);
        commasep(s, inconsistent, args, print_opt);
        pclose(s);
      }
      ast::expr_binary(op, lhs, rhs) {
        let prec = operator_prec(op);
        print_op_maybe_parens(s, lhs, prec);
        space(s.s);
        word_space(s, ast_util::binop_to_str(op));
        print_op_maybe_parens(s, rhs, prec + 1);
      }
      ast::expr_unary(op, expr) {
        word(s.s, ast_util::unop_to_str(op));
        print_op_maybe_parens(s, expr, parse::parser::unop_prec);
      }
      ast::expr_lit(lit) { print_literal(s, lit); }
      ast::expr_cast(expr, ty) {
        print_op_maybe_parens(s, expr, parse::parser::as_prec);
        space(s.s);
        word_space(s, "as");
        print_type(s, ty);
      }
      ast::expr_if(test, blk, elseopt) {
        print_if(s, test, blk, elseopt, false);
      }
      ast::expr_if_check(test, blk, elseopt) {
        print_if(s, test, blk, elseopt, true);
      }
      ast::expr_ternary(test, then, els) {
        print_expr(s, test);
        space(s.s);
        word_space(s, "?");
        print_expr(s, then);
        space(s.s);
        word_space(s, ":");
        print_expr(s, els);
      }
      ast::expr_while(test, blk) {
        head(s, "while");
        print_maybe_parens_discrim(s, test);
        space(s.s);
        print_block(s, blk);
      }
      ast::expr_for(decl, expr, blk) {
        head(s, "for");
        print_for_decl(s, decl, expr);
        space(s.s);
        print_block(s, blk);
      }
      ast::expr_for_each(decl, expr, blk) {
        head(s, "for each");
        print_for_decl(s, decl, expr);
        space(s.s);
        print_block(s, blk);
      }
      ast::expr_do_while(blk, expr) {
        head(s, "do");
        space(s.s);
        print_block(s, blk);
        space(s.s);
        word_space(s, "while");
        print_expr(s, expr);
      }
      ast::expr_alt(expr, arms) {
        cbox(s, alt_indent_unit);
        ibox(s, 4u);
        word_nbsp(s, "alt");
        print_maybe_parens_discrim(s, expr);
        space(s.s);
        bopen(s);
        for arm: ast::arm in arms {
            space(s.s);
            cbox(s, alt_indent_unit);
            ibox(s, 0u);
            let first = true;
            for p: @ast::pat in arm.pats {
                if first {
                    first = false;
                } else { space(s.s); word_space(s, "|"); }
                print_pat(s, p);
            }
            space(s.s);
            alt arm.guard {
              some(e) { word_space(s, "when"); print_expr(s, e); space(s.s); }
              none. { }
            }
            print_possibly_embedded_block(s, arm.body, block_normal,
                                          alt_indent_unit);
        }
        bclose_(s, expr.span, alt_indent_unit);
      }
      ast::expr_fn(f) {

        // If the return type is the magic ty_infer, then we need to
        // pretty print as a lambda-block
        if f.decl.output.node == ast::ty_infer {
            // containing cbox, will be closed by print-block at }
            cbox(s, indent_unit);
            // head-box, will be closed by print-block at start
            ibox(s, 0u);
            word(s.s, "{");
            print_fn_block_args(s, f.decl);
            print_possibly_embedded_block(s, f.body, block_block_fn,
                                          indent_unit);
        } else {
            head(s, proto_to_str(f.proto));
            print_fn_args_and_ret(s, f.decl, []);
            space(s.s);
            print_block(s, f.body);
        }
      }
      ast::expr_block(blk) {
        // containing cbox, will be closed by print-block at }
        cbox(s, indent_unit);
        // head-box, will be closed by print-block after {
        ibox(s, 0u);
        print_block(s, blk);
      }
      ast::expr_copy(e) { word_space(s, "copy"); print_expr(s, e); }
      ast::expr_move(lhs, rhs) {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, "<-");
        print_expr(s, rhs);
      }
      ast::expr_assign(lhs, rhs) {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, "=");
        print_expr(s, rhs);
      }
      ast::expr_swap(lhs, rhs) {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, "<->");
        print_expr(s, rhs);
      }
      ast::expr_assign_op(op, lhs, rhs) {
        print_expr(s, lhs);
        space(s.s);
        word(s.s, ast_util::binop_to_str(op));
        word_space(s, "=");
        print_expr(s, rhs);
      }
      ast::expr_field(expr, id) {
        // Deal with '10.x'
        if ends_in_lit_int(expr) {
            popen(s); print_expr(s, expr); pclose(s);
        } else {
            print_expr_parens_if_not_bot(s, expr);
        }
        word(s.s, ".");
        word(s.s, id);
      }
      ast::expr_index(expr, index) {
        print_expr_parens_if_not_bot(s, expr);
        word(s.s, "[");
        print_expr(s, index);
        word(s.s, "]");
      }
      ast::expr_path(path) { print_path(s, path, true); }
      ast::expr_fail(maybe_fail_val) {
        word(s.s, "fail");
        alt maybe_fail_val {
          some(expr) { word(s.s, " "); print_expr(s, expr); }
          _ { }
        }
      }
      ast::expr_break. { word(s.s, "break"); }
      ast::expr_cont. { word(s.s, "cont"); }
      ast::expr_ret(result) {
        word(s.s, "ret");
        alt result {
          some(expr) { word(s.s, " "); print_expr(s, expr); }
          _ { }
        }
      }
      ast::expr_put(result) {
        word(s.s, "put");
        alt result {
          some(expr) { word(s.s, " "); print_expr(s, expr); }
          _ { }
        }
      }
      ast::expr_be(result) { word_nbsp(s, "be"); print_expr(s, result); }
      ast::expr_log(lvl, expr) {
        alt lvl { 1 { word_nbsp(s, "log"); } 0 { word_nbsp(s, "log_err"); } }
        print_expr(s, expr);
      }
      ast::expr_check(m, expr) {
        alt m {
          ast::unchecked. { word_nbsp(s, "claim"); }
          ast::checked. { word_nbsp(s, "check"); }
        }
        popen(s);
        print_expr(s, expr);
        pclose(s);
      }
      ast::expr_assert(expr) {
        word_nbsp(s, "assert");
        popen(s);
        print_expr(s, expr);
        pclose(s);
      }
      ast::expr_mac(m) { print_mac(s, m); }
      ast::expr_anon_obj(anon_obj) {
        head(s, "obj");

        // Fields
        popen(s);
        fn print_field(s: ps, field: ast::anon_obj_field) {
            ibox(s, indent_unit);
            print_mutability(s, field.mut);
            word_space(s, field.ident + ":");
            print_type(s, field.ty);
            space(s.s);
            word_space(s, "=");
            print_expr(s, field.expr);
            end(s);
        }
        fn get_span(f: ast::anon_obj_field) -> codemap::span {
            ret f.ty.span;
        }
        alt anon_obj.fields {
          none. { }
          some(fields) {
            commasep_cmnt(s, consistent, fields, print_field, get_span);
          }
        }
        pclose(s);
        space(s.s);
        bopen(s);

        // Methods
        for meth: @ast::method in anon_obj.methods {
            let typarams: [ast::ty_param] = [];
            hardbreak_if_not_bol(s);
            maybe_print_comment(s, meth.span.lo);
            print_fn(s, meth.node.meth.decl, meth.node.meth.proto,
                     meth.node.ident, typarams, []);
            word(s.s, " ");
            print_block(s, meth.node.meth.body);
        }

        // With object
        alt anon_obj.inner_obj {
          none. { }
          some(e) { space(s.s); word_space(s, "with"); print_expr(s, e); }
        }
        bclose(s, expr.span);
      }
    }
    s.ann.post(ann_node);
    end(s);
}

fn print_expr_parens_if_not_bot(s: ps, ex: @ast::expr) {
    let parens = alt ex.node {
      ast::expr_fail(_) | ast::expr_ret(_) | ast::expr_put(_) |
      ast::expr_binary(_, _, _) | ast::expr_unary(_, _) |
      ast::expr_ternary(_, _, _) | ast::expr_move(_, _) |
      ast::expr_copy(_) | ast::expr_assign(_, _) | ast::expr_be(_) |
      ast::expr_assign_op(_, _, _) | ast::expr_swap(_, _) |
      ast::expr_log(_, _) | ast::expr_assert(_) |
      ast::expr_check(_, _) { true }
      _ { false }
    };
    if parens { popen(s); }
    print_expr(s, ex);
    if parens { pclose(s); }
}

fn print_local_decl(s: ps, loc: @ast::local) {
    print_pat(s, loc.node.pat);
    alt loc.node.ty.node {
      ast::ty_infer. { }
      _ { word_space(s, ":"); print_type(s, loc.node.ty); }
    }
}

fn print_decl(s: ps, decl: @ast::decl) {
    maybe_print_comment(s, decl.span.lo);
    alt decl.node {
      ast::decl_local(locs) {
        space_if_not_bol(s);
        ibox(s, indent_unit);
        word_nbsp(s, "let");
        fn print_local(s: ps, loc_st: (ast::let_style, @ast::local)) {
            let (st, loc) = loc_st;
            ibox(s, indent_unit);
            if st == ast::let_ref { word(s.s, "&"); }
            print_local_decl(s, loc);
            end(s);
            alt loc.node.init {
              some(init) {
                nbsp(s);
                alt init.op {
                  ast::init_assign. { word_space(s, "="); }
                  ast::init_move. { word_space(s, "<-"); }
                }
                print_expr(s, init.expr);
              }
              _ { }
            }
        }
        commasep(s, consistent, locs, print_local);
        end(s);
      }
      ast::decl_item(item) { print_item(s, item); }
    }
}

fn print_ident(s: ps, ident: ast::ident) { word(s.s, ident); }

fn print_for_decl(s: ps, loc: @ast::local, coll: @ast::expr) {
    print_local_decl(s, loc);
    space(s.s);
    word_space(s, "in");
    print_expr(s, coll);
}

fn print_path(s: ps, path: ast::path, colons_before_params: bool) {
    maybe_print_comment(s, path.span.lo);
    if path.node.global { word(s.s, "::"); }
    let first = true;
    for id: ast::ident in path.node.idents {
        if first { first = false; } else { word(s.s, "::"); }
        word(s.s, id);
    }
    if vec::len(path.node.types) > 0u {
        if colons_before_params { word(s.s, "::"); }
        word(s.s, "<");
        commasep(s, inconsistent, path.node.types, print_type);
        word(s.s, ">");
    }
}

fn print_pat(s: ps, pat: @ast::pat) {
    maybe_print_comment(s, pat.span.lo);
    let ann_node = node_pat(s, pat);
    s.ann.pre(ann_node);
    alt pat.node {
      ast::pat_wild. { word(s.s, "_"); }
      ast::pat_bind(id) { word(s.s, id); }
      ast::pat_lit(lit) { print_literal(s, lit); }
      ast::pat_tag(path, args) {
        print_path(s, path, true);
        if vec::len(args) > 0u {
            popen(s);
            commasep(s, inconsistent, args, print_pat);
            pclose(s);
        } else { word(s.s, "."); }
      }
      ast::pat_rec(fields, etc) {
        word(s.s, "{");
        fn print_field(s: ps, f: ast::field_pat) {
            cbox(s, indent_unit);
            word(s.s, f.ident);
            word_space(s, ":");
            print_pat(s, f.pat);
            end(s);
        }
        fn get_span(f: ast::field_pat) -> codemap::span { ret f.pat.span; }
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        if etc {
            if vec::len(fields) != 0u { word_space(s, ","); }
            word(s.s, "_");
        }
        word(s.s, "}");
      }
      ast::pat_tup(elts) {
        popen(s);
        commasep(s, inconsistent, elts, print_pat);
        pclose(s);
      }
      ast::pat_box(inner) { word(s.s, "@"); print_pat(s, inner); }
    }
    s.ann.post(ann_node);
}

fn print_fn(s: ps, decl: ast::fn_decl, proto: ast::proto, name: ast::ident,
            typarams: [ast::ty_param], constrs: [@ast::constr]) {
    alt decl.purity {
      ast::impure_fn. { head(s, proto_to_str(proto)); }
      _ { head(s, "pure fn"); }
    }
    word(s.s, name);
    print_type_params(s, typarams);
    print_fn_args_and_ret(s, decl, constrs);
}

fn print_fn_args_and_ret(s: ps, decl: ast::fn_decl, constrs: [@ast::constr]) {
    popen(s);
    fn print_arg(s: ps, x: ast::arg) {
        ibox(s, indent_unit);
        print_alias(s, x.mode);
        word_space(s, x.ident + ":");
        print_type(s, x.ty);
        end(s);
    }
    commasep(s, inconsistent, decl.inputs, print_arg);
    pclose(s);
    word(s.s, ast_fn_constrs_str(decl, constrs));
    maybe_print_comment(s, decl.output.span.lo);
    if decl.output.node != ast::ty_nil {
        space_if_not_bol(s);
        word_space(s, "->");
        alt decl.cf {
          ast::return_ref(mut, arg) {
            word(s.s, mut ? "&!" : "&");
            if vec::len(decl.inputs) > 1u {
                word(s.s, std::uint::str(arg));
            }
          }
          _ {}
        }
        print_type(s, decl.output);
    }
}

fn print_fn_block_args(s: ps, decl: ast::fn_decl) {
    word(s.s, "|");
    fn print_arg(s: ps, x: ast::arg) {
        ibox(s, indent_unit);
        print_alias(s, x.mode);
        word(s.s, x.ident);
        end(s);
    }
    commasep(s, inconsistent, decl.inputs, print_arg);
    word(s.s, "|");
    maybe_print_comment(s, decl.output.span.lo);
}

fn print_alias(s: ps, m: ast::mode) {
    alt m {
      ast::by_mut_ref. { word(s.s, "&"); }
      ast::by_move. { word(s.s, "-"); }
      ast::by_ref. { }
    }
}

fn print_kind(s: ps, kind: ast::kind) {
    alt kind {
      ast::kind_unique. { word(s.s, "~"); }
      ast::kind_shared. { word(s.s, "@"); }
      _ {/* fallthrough */ }
    }
}

fn print_type_params(s: ps, params: [ast::ty_param]) {
    if vec::len(params) > 0u {
        word(s.s, "<");
        fn printParam(s: ps, param: ast::ty_param) {
            print_kind(s, param.kind);
            word(s.s, param.ident);
        }
        commasep(s, inconsistent, params, printParam);
        word(s.s, ">");
    }
}

fn print_meta_item(s: ps, item: @ast::meta_item) {
    ibox(s, indent_unit);
    alt item.node {
      ast::meta_word(name) { word(s.s, name); }
      ast::meta_name_value(name, value) {
        word_space(s, name);
        word_space(s, "=");
        print_literal(s, @value);
      }
      ast::meta_list(name, items) {
        word(s.s, name);
        popen(s);
        commasep(s, consistent, items, print_meta_item);
        pclose(s);
      }
    }
    end(s);
}

fn print_view_item(s: ps, item: @ast::view_item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    alt item.node {
      ast::view_item_use(id, mta, _) {
        head(s, "use");
        word(s.s, id);
        if vec::len(mta) > 0u {
            popen(s);
            commasep(s, consistent, mta, print_meta_item);
            pclose(s);
        }
      }
      ast::view_item_import(id, ids, _) {
        head(s, "import");
        if !str::eq(id, ids[vec::len(ids) - 1u]) {
            word_space(s, id);
            word_space(s, "=");
        }
        let first = true;
        for elt: ast::ident in ids {
            if first { first = false; } else { word(s.s, "::"); }
            word(s.s, elt);
        }
      }
      ast::view_item_import_from(mod_path, idents, _) {
        head(s, "import");
        for elt: ast::ident in mod_path { word(s.s, elt); word(s.s, "::"); }
        word(s.s, "{");
        commasep(s, inconsistent, idents,
                 fn (s: ps, w: ast::import_ident) { word(s.s, w.node.name) });
        word(s.s, "}");
      }
      ast::view_item_import_glob(ids, _) {
        head(s, "import");
        let first = true;
        for elt: ast::ident in ids {
            if first { first = false; } else { word(s.s, "::"); }
            word(s.s, elt);
        }
        word(s.s, "::*");
      }
      ast::view_item_export(ids, _) {
        head(s, "export");
        commasep(s, inconsistent, ids,
                 fn (s: ps, w: ast::ident) { word(s.s, w) });
      }
    }
    word(s.s, ";");
    end(s); // end inner head-block

    end(s); // end outer head-block

}


// FIXME: The fact that this builds up the table anew for every call is
// not good. Eventually, table should be a const.
fn operator_prec(op: ast::binop) -> int {
    for spec: parse::parser::op_spec in *parse::parser::prec_table() {
        if spec.op == op { ret spec.prec; }
    }
    fail;
}

fn need_parens(expr: @ast::expr, outer_prec: int) -> bool {
    alt expr.node {
      ast::expr_binary(op, _, _) { operator_prec(op) < outer_prec }
      ast::expr_cast(_, _) { parse::parser::as_prec < outer_prec }
      ast::expr_ternary(_, _, _) { parse::parser::ternary_prec < outer_prec }





      // This may be too conservative in some cases
      ast::expr_assign(_, _) {
        true
      }
      ast::expr_move(_, _) { true }
      ast::expr_swap(_, _) { true }
      ast::expr_assign_op(_, _, _) { true }
      ast::expr_ret(_) { true }
      ast::expr_put(_) { true }
      ast::expr_be(_) { true }
      ast::expr_assert(_) { true }
      ast::expr_check(_, _) { true }
      ast::expr_log(_, _) { true }
      _ { false }
    }
}

fn print_op_maybe_parens(s: ps, expr: @ast::expr, outer_prec: int) {
    let add_them = need_parens(expr, outer_prec);
    if add_them { popen(s); }
    print_expr(s, expr);
    if add_them { pclose(s); }
}

fn print_mutability(s: ps, mut: ast::mutability) {
    alt mut {
      ast::mut. { word_nbsp(s, "mutable"); }
      ast::maybe_mut. { word_nbsp(s, "mutable?"); }
      ast::imm. {/* nothing */ }
    }
}

fn print_mt(s: ps, mt: ast::mt) {
    print_mutability(s, mt.mut);
    print_type(s, mt.ty);
}

fn print_ty_fn(s: ps, proto: ast::proto, id: option::t<ast::ident>,
               inputs: [ast::ty_arg], output: @ast::ty, cf: ast::ret_style,
               constrs: [@ast::constr]) {
    ibox(s, indent_unit);
    word(s.s, proto_to_str(proto));
    alt id { some(id) { word(s.s, " "); word(s.s, id); } _ { } }
    zerobreak(s.s);
    popen(s);
    fn print_arg(s: ps, input: ast::ty_arg) {
        print_alias(s, input.node.mode);
        print_type(s, input.node.ty);
    }
    commasep(s, inconsistent, inputs, print_arg);
    pclose(s);
    maybe_print_comment(s, output.span.lo);
    if output.node != ast::ty_nil {
        space_if_not_bol(s);
        ibox(s, indent_unit);
        word_space(s, "->");
        if cf == ast::noreturn {
            word_nbsp(s, "!");
        } else {
            alt cf {
              ast::return_ref(mut, arg) {
                word(s.s, mut ? "&!" : "&");
                if vec::len(inputs) > 1u { word(s.s, std::uint::str(arg)); }
              }
              _ {}
            }
            print_type(s, output);
        }
        end(s);
    }
    word(s.s, ast_ty_fn_constrs_str(constrs));
    end(s);
}

fn maybe_print_trailing_comment(s: ps, span: codemap::span,
                                next_pos: option::t<uint>) {
    let cm;
    alt s.cm { some(ccm) { cm = ccm; } _ { ret; } }
    alt next_comment(s) {
      some(cmnt) {
        if cmnt.style != lexer::trailing { ret; }
        let span_line = codemap::lookup_char_pos(cm, span.hi);
        let comment_line = codemap::lookup_char_pos(cm, cmnt.pos);
        let next = cmnt.pos + 1u;
        alt next_pos { none. { } some(p) { next = p; } }
        if span.hi < cmnt.pos && cmnt.pos < next &&
               span_line.line == comment_line.line {
            print_comment(s, cmnt);
            s.cur_cmnt += 1u;
        }
      }
      _ { }
    }
}

fn print_remaining_comments(s: ps) {
    // If there aren't any remaining comments, then we need to manually
    // make sure there is a line break at the end.
    if option::is_none(next_comment(s)) { hardbreak(s.s); }
    while true {
        alt next_comment(s) {
          some(cmnt) { print_comment(s, cmnt); s.cur_cmnt += 1u; }
          _ { break; }
        }
    }
}

fn in_cbox(s: ps) -> bool {
    let len = vec::len(s.boxes);
    if len == 0u { ret false; }
    ret s.boxes[len - 1u] == pp::consistent;
}

fn print_literal(s: ps, lit: @ast::lit) {
    maybe_print_comment(s, lit.span.lo);
    alt next_lit(s) {
      some(lt) {
        if lt.pos == lit.span.lo { word(s.s, lt.lit); s.cur_lit += 1u; ret; }
      }
      _ { }
    }
    alt lit.node {
      ast::lit_str(st) { print_string(s, st); }
      ast::lit_char(ch) {
        word(s.s,
             "'" + escape_str(str::unsafe_from_bytes([ch as u8]), '\'') +
                 "'");
      }
      ast::lit_int(val) { word(s.s, int::str(val)); }
      ast::lit_uint(val) { word(s.s, uint::str(val) + "u"); }
      ast::lit_float(fstr) { word(s.s, fstr); }
      ast::lit_mach_int(mach, val) {
        word(s.s, int::str(val as int));
        word(s.s, ast_util::ty_mach_to_str(mach));
      }
      ast::lit_mach_float(mach, val) {
        // val is already a str
        word(s.s, val);
        word(s.s, ast_util::ty_mach_to_str(mach));
      }
      ast::lit_nil. { word(s.s, "()"); }
      ast::lit_bool(val) {
        if val { word(s.s, "true"); } else { word(s.s, "false"); }
      }
    }
}

fn lit_to_str(l: @ast::lit) -> str { be to_str(l, print_literal); }

fn next_lit(s: ps) -> option::t<lexer::lit> {
    alt s.literals {
      some(lits) {
        if s.cur_lit < vec::len(lits) {
            ret some(lits[s.cur_lit]);
        } else { ret none::<lexer::lit>; }
      }
      _ { ret none::<lexer::lit>; }
    }
}

fn maybe_print_comment(s: ps, pos: uint) {
    while true {
        alt next_comment(s) {
          some(cmnt) {
            if cmnt.pos < pos {
                print_comment(s, cmnt);
                s.cur_cmnt += 1u;
            } else { break; }
          }
          _ { break; }
        }
    }
}

fn print_comment(s: ps, cmnt: lexer::cmnt) {
    alt cmnt.style {
      lexer::mixed. {
        assert (vec::len(cmnt.lines) == 1u);
        zerobreak(s.s);
        word(s.s, cmnt.lines[0]);
        zerobreak(s.s);
      }
      lexer::isolated. {
        pprust::hardbreak_if_not_bol(s);
        for line: str in cmnt.lines {
            // Don't print empty lines because they will end up as trailing
            // whitespace
            if str::is_not_empty(line) { word(s.s, line); }
            hardbreak(s.s);
        }
      }
      lexer::trailing. {
        word(s.s, " ");
        if vec::len(cmnt.lines) == 1u {
            word(s.s, cmnt.lines[0]);
            hardbreak(s.s);
        } else {
            ibox(s, 0u);
            for line: str in cmnt.lines {
                if str::is_not_empty(line) { word(s.s, line); }
                hardbreak(s.s);
            }
            end(s);
        }
      }
      lexer::blank_line. {
        // We need to do at least one, possibly two hardbreaks.
        let is_semi =
            alt s.s.last_token() {
              pp::STRING(s, _) { s == ";" }
              _ { false }
            };
        if is_semi || is_begin(s) || is_end(s) { hardbreak(s.s); }
        hardbreak(s.s);
      }
    }
}

fn print_string(s: ps, st: str) {
    word(s.s, "\"");
    word(s.s, escape_str(st, '"'));
    word(s.s, "\"");
}

fn escape_str(st: str, to_escape: char) -> str {
    let out: str = "";
    let len = str::byte_len(st);
    let i = 0u;
    while i < len {
        alt st[i] as char {
          '\n' { out += "\\n"; }
          '\t' { out += "\\t"; }
          '\r' { out += "\\r"; }
          '\\' { out += "\\\\"; }
          cur {
            if cur == to_escape { out += "\\"; }
            // FIXME some (or all?) non-ascii things should be escaped

            str::push_char(out, cur);
          }
        }
        i += 1u;
    }
    ret out;
}

fn to_str<T>(t: T, f: fn(ps, T)) -> str {
    let writer = io::string_writer();
    let s = rust_printer(writer.get_writer());
    f(s, t);
    eof(s.s);
    ret writer.get_str();
}

fn next_comment(s: ps) -> option::t<lexer::cmnt> {
    alt s.comments {
      some(cmnts) {
        if s.cur_cmnt < vec::len(cmnts) {
            ret some(cmnts[s.cur_cmnt]);
        } else { ret none::<lexer::cmnt>; }
      }
      _ { ret none::<lexer::cmnt>; }
    }
}

// Removing the aliases from the type of f in the next two functions
// triggers memory corruption, but I haven't isolated the bug yet. FIXME
fn constr_args_to_str<T>(f: fn(T) -> str, args: [@ast::sp_constr_arg<T>]) ->
   str {
    let comma = false;
    let s = "(";
    for a: @ast::sp_constr_arg<T> in args {
        if comma { s += ", "; } else { comma = true; }
        s += constr_arg_to_str::<T>(f, a.node);
    }
    s += ")";
    ret s;
}

fn constr_arg_to_str<T>(f: fn(T) -> str, c: ast::constr_arg_general_<T>) ->
   str {
    alt c {
      ast::carg_base. { ret "*"; }
      ast::carg_ident(i) { ret f(i); }
      ast::carg_lit(l) { ret lit_to_str(l); }
    }
}

// needed b/c constr_args_to_str needs
// something that takes an alias
// (argh)
fn uint_to_str(i: uint) -> str { ret uint::str(i); }

fn ast_ty_fn_constr_to_str(c: @ast::constr) -> str {
    ret path_to_str(c.node.path) +
            constr_args_to_str(uint_to_str, c.node.args);
}

// FIXME: fix repeated code
fn ast_ty_fn_constrs_str(constrs: [@ast::constr]) -> str {
    let s = "";
    let colon = true;
    for c: @ast::constr in constrs {
        if colon { s += " : "; colon = false; } else { s += ", "; }
        s += ast_ty_fn_constr_to_str(c);
    }
    ret s;
}

fn fn_arg_idx_to_str(decl: ast::fn_decl, idx: uint) -> str {
    decl.inputs[idx].ident
}

fn ast_fn_constr_to_str(decl: ast::fn_decl, c: @ast::constr) -> str {
    let arg_to_str = bind fn_arg_idx_to_str(decl, _);
    ret path_to_str(c.node.path) +
            constr_args_to_str(arg_to_str, c.node.args);
}

// FIXME: fix repeated code
fn ast_fn_constrs_str(decl: ast::fn_decl, constrs: [@ast::constr]) -> str {
    let s = "";
    let colon = true;
    for c: @ast::constr in constrs {
        if colon { s += " : "; colon = false; } else { s += ", "; }
        s += ast_fn_constr_to_str(decl, c);
    }
    ret s;
}

fn proto_to_str(p: ast::proto) -> str {
    ret alt p {
          ast::proto_fn. { "fn" }
          ast::proto_iter. { "iter" }
          ast::proto_block. { "block" }
          ast::proto_closure. { "lambda" }
        };
}

fn ty_constr_to_str(c: @ast::ty_constr) -> str {
    fn ty_constr_path_to_str(p: ast::path) -> str { "*." + path_to_str(p) }

    ret path_to_str(c.node.path) +
            constr_args_to_str::<ast::path>(ty_constr_path_to_str,
                                            c.node.args);
}


fn ast_ty_constrs_str(constrs: [@ast::ty_constr]) -> str {
    let s = "";
    let colon = true;
    for c: @ast::ty_constr in constrs {
        if colon { s += " : "; colon = false; } else { s += ", "; }
        s += ty_constr_to_str(c);
    }
    ret s;
}

fn ends_in_lit_int(ex: @ast::expr) -> bool {
    alt ex.node {
      ast::expr_lit(@{node: ast::lit_int(_), _}) { true }
      ast::expr_binary(_, _, sub) | ast::expr_unary(_, sub) |
      ast::expr_ternary(_, _, sub) | ast::expr_move(_, sub) |
      ast::expr_copy(sub) | ast::expr_assign(_, sub) | ast::expr_be(sub) |
      ast::expr_assign_op(_, _, sub) | ast::expr_swap(_, sub) |
      ast::expr_log(_, sub) | ast::expr_assert(sub) |
      ast::expr_check(_, sub) { ends_in_lit_int(sub) }
      ast::expr_fail(osub) | ast::expr_ret(osub) | ast::expr_put(osub) {
        alt osub {
          some(ex) { ends_in_lit_int(ex) }
          _ { false }
        }
      }
      _ { false }
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
