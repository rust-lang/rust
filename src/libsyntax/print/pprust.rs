use parse::{comments, lexer, token};
use codemap::codemap;
use pp::{break_offset, word, printer, space, zerobreak, hardbreak, breaks};
use pp::{consistent, inconsistent, eof};
use ast::{required, provided};
use ast_util::{operator_prec};
use dvec::DVec;
use parse::classify::*;
use parse::token::ident_interner;

// The ps is stored here to prevent recursive type.
enum ann_node {
    node_block(ps, ast::blk),
    node_item(ps, @ast::item),
    node_expr(ps, @ast::expr),
    node_pat(ps, @ast::pat),
}
type pp_ann = {pre: fn@(ann_node), post: fn@(ann_node)};

fn no_ann() -> pp_ann {
    fn ignore(_node: ann_node) { }
    return {pre: ignore, post: ignore};
}

type ps =
    @{s: pp::printer,
      cm: Option<codemap>,
      intr: token::ident_interner,
      comments: Option<~[comments::cmnt]>,
      literals: Option<~[comments::lit]>,
      mut cur_cmnt: uint,
      mut cur_lit: uint,
      boxes: DVec<pp::breaks>,
      ann: pp_ann};

fn ibox(s: ps, u: uint) {
    s.boxes.push(pp::inconsistent);
    pp::ibox(s.s, u);
}

fn end(s: ps) {
    s.boxes.pop();
    pp::end(s.s);
}

fn rust_printer(writer: io::Writer, intr: ident_interner) -> ps {
    return @{s: pp::mk_printer(writer, default_columns),
             cm: None::<codemap>,
             intr: intr,
             comments: None::<~[comments::cmnt]>,
             literals: None::<~[comments::lit]>,
             mut cur_cmnt: 0u,
             mut cur_lit: 0u,
             boxes: DVec(),
             ann: no_ann()};
}

const indent_unit: uint = 4u;
const alt_indent_unit: uint = 2u;

const default_columns: uint = 78u;

// Requires you to pass an input filename and reader so that
// it can scan the input text for comments and literals to
// copy forward.
fn print_crate(cm: codemap, intr: ident_interner,
               span_diagnostic: diagnostic::span_handler,
               crate: @ast::crate, filename: ~str, in: io::Reader,
               out: io::Writer, ann: pp_ann, is_expanded: bool) {
    let r = comments::gather_comments_and_literals(span_diagnostic,
                                                   filename, in);
    let s =
        @{s: pp::mk_printer(out, default_columns),
          cm: Some(cm),
          intr: intr,
          comments: Some(r.cmnts),
          // If the code is post expansion, don't use the table of
          // literals, since it doesn't correspond with the literals
          // in the AST anymore.
          literals: if is_expanded { None } else { Some(r.lits) },
          mut cur_cmnt: 0u,
          mut cur_lit: 0u,
          boxes: DVec(),
          ann: ann};
    print_crate_(s, crate);
}

fn print_crate_(s: ps, &&crate: @ast::crate) {
    print_mod(s, crate.node.module, crate.node.attrs);
    print_remaining_comments(s);
    eof(s.s);
}

fn ty_to_str(ty: @ast::ty, intr: ident_interner) -> ~str {
    to_str(ty, print_type, intr)
}

fn pat_to_str(pat: @ast::pat, intr: ident_interner) -> ~str {
    to_str(pat, print_pat, intr)
}

fn expr_to_str(e: @ast::expr, intr: ident_interner) -> ~str {
    to_str(e, print_expr, intr)
}

fn tt_to_str(tt: ast::token_tree, intr: ident_interner) -> ~str {
    to_str(tt, print_tt, intr)
}

fn stmt_to_str(s: ast::stmt, intr: ident_interner) -> ~str {
    to_str(s, print_stmt, intr)
}

fn item_to_str(i: @ast::item, intr: ident_interner) -> ~str {
    to_str(i, print_item, intr)
}

fn typarams_to_str(tps: ~[ast::ty_param], intr: ident_interner) -> ~str {
    to_str(tps, print_type_params, intr)
}

fn path_to_str(&&p: @ast::path, intr: ident_interner) -> ~str {
    to_str(p, |a,b| print_path(a, b, false), intr)
}

fn fun_to_str(decl: ast::fn_decl, name: ast::ident,
              params: ~[ast::ty_param], intr: ident_interner) -> ~str {
    let buffer = io::mem_buffer();
    let s = rust_printer(io::mem_buffer_writer(buffer), intr);
    print_fn(s, decl, None, name, params, None);
    end(s); // Close the head box
    end(s); // Close the outer box
    eof(s.s);
    io::mem_buffer_str(buffer)
}

#[test]
fn test_fun_to_str() {
    let decl: ast::fn_decl = {
        inputs: ~[],
        output: @{id: 0,
                  node: ast::ty_nil,
                  span: ast_util::dummy_sp()},
        purity: ast::impure_fn,
        cf: ast::return_val
    };
    assert fun_to_str(decl, "a", ~[]) == "fn a()";
}

fn block_to_str(blk: ast::blk, intr: ident_interner) -> ~str {
    let buffer = io::mem_buffer();
    let s = rust_printer(io::mem_buffer_writer(buffer), intr);
    // containing cbox, will be closed by print-block at }
    cbox(s, indent_unit);
    // head-ibox, will be closed by print-block after {
    ibox(s, 0u);
    print_block(s, blk);
    eof(s.s);
    io::mem_buffer_str(buffer)
}

fn meta_item_to_str(mi: @ast::meta_item, intr: ident_interner) -> ~str {
    to_str(mi, print_meta_item, intr)
}

fn attribute_to_str(attr: ast::attribute, intr: ident_interner) -> ~str {
    to_str(attr, print_attribute, intr)
}

fn variant_to_str(var: ast::variant, intr: ident_interner) -> ~str {
    to_str(var, print_variant, intr)
}

#[test]
fn test_variant_to_str() {
    let var = ast_util::respan(ast_util::dummy_sp(), {
        name: "principle_skinner",
        attrs: ~[],
        args: ~[],
        id: 0,
        disr_expr: none
    });

    let varstr = variant_to_str(var);
    assert varstr == "principle_skinner";
}

fn cbox(s: ps, u: uint) {
    s.boxes.push(pp::consistent);
    pp::cbox(s.s, u);
}

fn box(s: ps, u: uint, b: pp::breaks) {
    s.boxes.push(b);
    pp::box(s.s, u, b);
}

fn nbsp(s: ps) { word(s.s, ~" "); }

fn word_nbsp(s: ps, w: ~str) { word(s.s, w); nbsp(s); }

fn word_space(s: ps, w: ~str) { word(s.s, w); space(s.s); }

fn popen(s: ps) { word(s.s, ~"("); }

fn pclose(s: ps) { word(s.s, ~")"); }

fn head(s: ps, w: ~str) {
    // outer-box is consistent
    cbox(s, indent_unit);
    // head-box is inconsistent
    ibox(s, str::len(w) + 1);
    // keyword that starts the head
    word_nbsp(s, w);
}

fn bopen(s: ps) {
    word(s.s, ~"{");
    end(s); // close the head-box
}

fn bclose_(s: ps, span: codemap::span, indented: uint) {
    bclose_maybe_open(s, span, indented, true);
}
fn bclose_maybe_open (s: ps, span: codemap::span, indented: uint,
                     close_box: bool) {
    maybe_print_comment(s, span.hi);
    break_offset_if_not_bol(s, 1u, -(indented as int));
    word(s.s, ~"}");
    if close_box {
        end(s); // close the outer-box
    }
}
fn bclose(s: ps, span: codemap::span) { bclose_(s, span, indent_unit); }

fn is_begin(s: ps) -> bool {
    match s.s.last_token() { pp::BEGIN(_) => true, _ => false }
}

fn is_end(s: ps) -> bool {
    match s.s.last_token() { pp::END => true, _ => false }
}

fn is_bol(s: ps) -> bool {
    return s.s.last_token().is_eof() || s.s.last_token().is_hardbreak_tok();
}

fn in_cbox(s: ps) -> bool {
    let len = s.boxes.len();
    if len == 0u { return false; }
    return s.boxes[len - 1u] == pp::consistent;
}

fn hardbreak_if_not_bol(s: ps) { if !is_bol(s) { hardbreak(s.s); } }
fn space_if_not_bol(s: ps) { if !is_bol(s) { space(s.s); } }
fn break_offset_if_not_bol(s: ps, n: uint, off: int) {
    if !is_bol(s) {
        break_offset(s.s, n, off);
    } else {
        if off != 0 && s.s.last_token().is_hardbreak_tok() {
            // We do something pretty sketchy here: tuck the nonzero
            // offset-adjustment we were going to deposit along with the
            // break into the previous hardbreak.
            s.s.replace_last_token(pp::hardbreak_tok_offset(off));
        }
    }
}

// Synthesizes a comment that was not textually present in the original source
// file.
fn synth_comment(s: ps, text: ~str) {
    word(s.s, ~"/*");
    space(s.s);
    word(s.s, text);
    space(s.s);
    word(s.s, ~"*/");
}

fn commasep<IN>(s: ps, b: breaks, elts: ~[IN], op: fn(ps, IN)) {
    box(s, 0u, b);
    let mut first = true;
    for elts.each |elt| {
        if first { first = false; } else { word_space(s, ~","); }
        op(s, elt);
    }
    end(s);
}


fn commasep_cmnt<IN>(s: ps, b: breaks, elts: ~[IN], op: fn(ps, IN),
                     get_span: fn(IN) -> codemap::span) {
    box(s, 0u, b);
    let len = vec::len::<IN>(elts);
    let mut i = 0u;
    for elts.each |elt| {
        maybe_print_comment(s, get_span(elt).hi);
        op(s, elt);
        i += 1u;
        if i < len {
            word(s.s, ~",");
            maybe_print_trailing_comment(s, get_span(elt),
                                         Some(get_span(elts[i]).hi));
            space_if_not_bol(s);
        }
    }
    end(s);
}

fn commasep_exprs(s: ps, b: breaks, exprs: ~[@ast::expr]) {
    fn expr_span(&&expr: @ast::expr) -> codemap::span { return expr.span; }
    commasep_cmnt(s, b, exprs, print_expr, expr_span);
}

fn print_mod(s: ps, _mod: ast::_mod, attrs: ~[ast::attribute]) {
    print_inner_attributes(s, attrs);
    for _mod.view_items.each |vitem| {
        print_view_item(s, vitem);
    }
    for _mod.items.each |item| { print_item(s, item); }
}

fn print_foreign_mod(s: ps, nmod: ast::foreign_mod,
                     attrs: ~[ast::attribute]) {
    print_inner_attributes(s, attrs);
    for nmod.view_items.each |vitem| {
        print_view_item(s, vitem);
    }
    for nmod.items.each |item| { print_foreign_item(s, item); }
}

fn print_region(s: ps, region: @ast::region) {
    match region.node {
      ast::re_anon => word_space(s, ~"&"),
      ast::re_named(name) => {
        word(s.s, ~"&");
        print_ident(s, name);
      }
    }
}

fn print_type(s: ps, &&ty: @ast::ty) {
    print_type_ex(s, ty, false);
}

fn print_type_ex(s: ps, &&ty: @ast::ty, print_colons: bool) {
    maybe_print_comment(s, ty.span.lo);
    ibox(s, 0u);
    match ty.node {
      ast::ty_nil => word(s.s, ~"()"),
      ast::ty_bot => word(s.s, ~"!"),
      ast::ty_box(mt) => { word(s.s, ~"@"); print_mt(s, mt); }
      ast::ty_uniq(mt) => { word(s.s, ~"~"); print_mt(s, mt); }
      ast::ty_vec(mt) => {
        word(s.s, ~"[");
        match mt.mutbl {
          ast::m_mutbl => word_space(s, ~"mut"),
          ast::m_const => word_space(s, ~"const"),
          ast::m_imm => ()
        }
        print_type(s, mt.ty);
        word(s.s, ~"]");
      }
      ast::ty_ptr(mt) => { word(s.s, ~"*"); print_mt(s, mt); }
      ast::ty_rptr(region, mt) => {
        match region.node {
          ast::re_anon => word(s.s, ~"&"),
          _ => { print_region(s, region); word(s.s, ~"/"); }
        }
        print_mt(s, mt);
      }
      ast::ty_rec(fields) => {
        word(s.s, ~"{");
        fn print_field(s: ps, f: ast::ty_field) {
            cbox(s, indent_unit);
            print_mutability(s, f.node.mt.mutbl);
            print_ident(s, f.node.ident);
            word_space(s, ~":");
            print_type(s, f.node.mt.ty);
            end(s);
        }
        fn get_span(f: ast::ty_field) -> codemap::span { return f.span; }
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        word(s.s, ~",}");
      }
      ast::ty_tup(elts) => {
        popen(s);
        commasep(s, inconsistent, elts, print_type);
        pclose(s);
      }
      ast::ty_fn(proto, purity, bounds, d) => {
        print_ty_fn(s, Some(proto), purity, bounds, d, None, None, None);
      }
      ast::ty_path(path, _) => print_path(s, path, print_colons),
      ast::ty_fixed_length(t, v) => {
        print_type(s, t);
        word(s.s, ~"/");
        print_vstore(s, ast::vstore_fixed(v));
      }
      ast::ty_mac(_) => {
          fail ~"print_type doesn't know how to print a ty_mac";
      }
      ast::ty_infer => {
          fail ~"print_type shouldn't see a ty_infer";
      }

    }
    end(s);
}

fn print_foreign_item(s: ps, item: @ast::foreign_item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    match item.node {
      ast::foreign_item_fn(decl, purity, typarams) => {
        print_fn(s, decl, Some(purity), item.ident, typarams, None);
        end(s); // end head-ibox
        word(s.s, ~";");
        end(s); // end the outer fn box
      }
      ast::foreign_item_const(t) => {
        head(s, ~"const");
        print_ident(s, item.ident);
        word_space(s, ~":");
        print_type(s, t);
        word(s.s, ~";");
        end(s); // end the head-ibox
      }
    }
}

fn print_item(s: ps, &&item: @ast::item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    let ann_node = node_item(s, item);
    s.ann.pre(ann_node);
    match item.node {
      ast::item_const(ty, expr) => {
        head(s, ~"const");
        print_ident(s, item.ident);
        word_space(s, ~":");
        print_type(s, ty);
        space(s.s);
        end(s); // end the head-ibox

        word_space(s, ~"=");
        print_expr(s, expr);
        word(s.s, ~";");
        end(s); // end the outer cbox

      }
      ast::item_fn(decl, purity, typarams, body) => {
        print_fn(s, decl, Some(purity), item.ident, typarams, None);
        word(s.s, ~" ");
        print_block_with_attrs(s, body, item.attrs);
      }
      ast::item_mod(_mod) => {
        head(s, ~"mod");
        print_ident(s, item.ident);
        nbsp(s);
        bopen(s);
        print_mod(s, _mod, item.attrs);
        bclose(s, item.span);
      }
      ast::item_foreign_mod(nmod) => {
        head(s, ~"extern");
        match nmod.sort {
            ast::named => {
                word_nbsp(s, ~"mod");
                print_ident(s, item.ident);
            }
            ast::anonymous => {}
        }
        nbsp(s);
        bopen(s);
        print_foreign_mod(s, nmod, item.attrs);
        bclose(s, item.span);
      }
      ast::item_ty(ty, params) => {
        ibox(s, indent_unit);
        ibox(s, 0u);
        word_nbsp(s, ~"type");
        print_ident(s, item.ident);
        print_type_params(s, params);
        end(s); // end the inner ibox

        space(s.s);
        word_space(s, ~"=");
        print_type(s, ty);
        word(s.s, ~";");
        end(s); // end the outer ibox
      }
      ast::item_enum(enum_definition, params) => {
        print_enum_def(s, enum_definition, params, item.ident, item.span);
      }
      ast::item_class(struct_def, tps) => {
          head(s, ~"struct");
          print_struct(s, struct_def, tps, item.ident, item.span);
      }

      ast::item_impl(tps, opt_trait, ty, methods) => {
        head(s, ~"impl");
        if tps.is_not_empty() {
            print_type_params(s, tps);
            space(s.s);
        }
        print_type(s, ty);

        match opt_trait {
            Some(t) => {
                word_space(s, ~":");
                print_path(s, t.path, false);
            }
            None => ()
        };
        space(s.s);

        bopen(s);
        for methods.each |meth| {
           print_method(s, meth);
        }
        bclose(s, item.span);
      }
      ast::item_trait(tps, traits, methods) => {
        head(s, ~"trait");
        print_ident(s, item.ident);
        print_type_params(s, tps);
        if vec::len(traits) != 0u {
            word_space(s, ~":");
            commasep(s, inconsistent, traits, |s, p|
                print_path(s, p.path, false));
        }
        word(s.s, ~" ");
        bopen(s);
        for methods.each |meth| { print_trait_method(s, meth); }
        bclose(s, item.span);
      }
      ast::item_mac({node: ast::mac_invoc_tt(pth, tts), _}) => {
        print_path(s, pth, false);
        word(s.s, ~"! ");
        print_ident(s, item.ident);
        cbox(s, indent_unit);
        popen(s);
        for tts.each |tt| { print_tt(s, tt);  }
        pclose(s);
        end(s);
      }
      ast::item_mac(_) => {
        fail ~"invalid item-position syntax bit"
      }
    }
    s.ann.post(ann_node);
}

fn print_enum_def(s: ps, enum_definition: ast::enum_def,
                  params: ~[ast::ty_param], ident: ast::ident,
                  span: ast::span) {
    let mut newtype =
        vec::len(enum_definition.variants) == 1u &&
        ident == enum_definition.variants[0].node.name;
    if newtype {
        match enum_definition.variants[0].node.kind {
            ast::tuple_variant_kind(args) if args.len() == 1 => {}
            _ => newtype = false
        }
    }
    if newtype {
        ibox(s, indent_unit);
        word_space(s, ~"enum");
    } else {
        head(s, ~"enum");
    }

    print_ident(s, ident);
    print_type_params(s, params);
    space(s.s);
    if newtype {
        word_space(s, ~"=");
        match enum_definition.variants[0].node.kind {
            ast::tuple_variant_kind(args) => print_type(s, args[0].ty),
            _ => fail ~"newtype syntax with struct?"
        }
        word(s.s, ~";");
        end(s);
    } else {
        print_variants(s, enum_definition.variants, span);
    }
}

fn print_variants(s: ps, variants: ~[ast::variant], span: ast::span) {
    bopen(s);
    for variants.each |v| {
        space_if_not_bol(s);
        maybe_print_comment(s, v.span.lo);
        print_outer_attributes(s, v.node.attrs);
        ibox(s, indent_unit);
        print_variant(s, v);
        word(s.s, ~",");
        end(s);
        maybe_print_trailing_comment(s, v.span, None::<uint>);
    }
    bclose(s, span);
}

fn print_struct(s: ps, struct_def: @ast::struct_def, tps: ~[ast::ty_param],
                ident: ast::ident, span: ast::span) {
    print_ident(s, ident);
    nbsp(s);
    print_type_params(s, tps);
    if vec::len(struct_def.traits) != 0u {
        word_space(s, ~":");
        commasep(s, inconsistent, struct_def.traits, |s, p|
            print_path(s, p.path, false));
    }
    bopen(s);
    hardbreak_if_not_bol(s);
    do option::iter(struct_def.ctor) |ctor| {
      maybe_print_comment(s, ctor.span.lo);
      print_outer_attributes(s, ctor.node.attrs);
      // Doesn't call head because there shouldn't be a space after new.
      cbox(s, indent_unit);
      ibox(s, 4);
      word(s.s, ~"new(");
      print_fn_args(s, ctor.node.dec, ~[], None);
      word(s.s, ~")");
      space(s.s);
      print_block(s, ctor.node.body);
    }
    do option::iter(struct_def.dtor) |dtor| {
      hardbreak_if_not_bol(s);
      maybe_print_comment(s, dtor.span.lo);
      print_outer_attributes(s, dtor.node.attrs);
      head(s, ~"drop");
      print_block(s, dtor.node.body);
    }
    for struct_def.fields.each |field| {
        match field.node.kind {
            ast::unnamed_field => {} // We don't print here.
            ast::named_field(ident, mutability, visibility) => {
                hardbreak_if_not_bol(s);
                maybe_print_comment(s, field.span.lo);
                if visibility == ast::private {
                    word_nbsp(s, ~"priv");
                }
                if mutability == ast::class_mutable {
                    word_nbsp(s, ~"mut");
                }
                print_ident(s, ident);
                word_nbsp(s, ~":");
                print_type(s, field.node.ty);
                word(s.s, ~",");
            }
        }
    }
    for struct_def.methods.each |method| {
        print_method(s, method);
    }
    bclose(s, span);
}

/// This doesn't deserve to be called "pretty" printing, but it should be
/// meaning-preserving. A quick hack that might help would be to look at the
/// spans embedded in the TTs to decide where to put spaces and newlines.
/// But it'd be better to parse these according to the grammar of the
/// appropriate macro, transcribe back into the grammar we just parsed from,
/// and then pretty-print the resulting AST nodes (so, e.g., we print
/// expression arguments as expressions). It can be done! I think.
fn print_tt(s: ps, tt: ast::token_tree) {
    match tt {
      ast::tt_delim(tts) => for tts.each() |tt_elt| { print_tt(s, tt_elt); },
      ast::tt_tok(_, tk) => {
        match tk {
          parse::token::IDENT(*) => { // don't let idents run together
            if s.s.token_tree_last_was_ident { word(s.s, ~" ") }
            s.s.token_tree_last_was_ident = true;
          }
          _ => { s.s.token_tree_last_was_ident = false; }
        }
        word(s.s, parse::token::to_str(s.intr, tk));
      }
      ast::tt_seq(_, tts, sep, zerok) => {
        word(s.s, ~"$(");
        for tts.each() |tt_elt| { print_tt(s, tt_elt); }
        word(s.s, ~")");
        match sep {
          Some(tk) => word(s.s, parse::token::to_str(s.intr, tk)),
          None => ()
        }
        word(s.s, if zerok { ~"*" } else { ~"+" });
        s.s.token_tree_last_was_ident = false;
      }
      ast::tt_nonterminal(_, name) => {
        word(s.s, ~"$");
        print_ident(s, name);
        s.s.token_tree_last_was_ident = true;
      }
    }
}

fn print_variant(s: ps, v: ast::variant) {
    match v.node.kind {
        ast::tuple_variant_kind(args) => {
            print_ident(s, v.node.name);
            if vec::len(args) > 0u {
                popen(s);
                fn print_variant_arg(s: ps, arg: ast::variant_arg) {
                    print_type(s, arg.ty);
                }
                commasep(s, consistent, args, print_variant_arg);
                pclose(s);
            }
        }
        ast::struct_variant_kind(struct_def) => {
            head(s, ~"");
            print_struct(s, struct_def, ~[], v.node.name, v.span);
        }
        ast::enum_variant_kind(enum_definition) => {
            print_variants(s, enum_definition.variants, v.span);
        }
    }
    match v.node.disr_expr {
      Some(d) => {
        space(s.s);
        word_space(s, ~"=");
        print_expr(s, d);
      }
      _ => ()
    }
}

fn print_ty_method(s: ps, m: ast::ty_method) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, m.span.lo);
    print_outer_attributes(s, m.attrs);
    print_ty_fn(s, None, m.purity,
                @~[], m.decl, Some(m.ident), Some(m.tps),
                Some(m.self_ty.node));
    word(s.s, ~";");
}

fn print_trait_method(s: ps, m: ast::trait_method) {
    match m {
      required(ty_m) => print_ty_method(s, ty_m),
      provided(m)    => print_method(s, m)
    }
}

fn print_method(s: ps, meth: @ast::method) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, meth.span.lo);
    print_outer_attributes(s, meth.attrs);
    print_fn(s, meth.decl, Some(meth.purity),
             meth.ident, meth.tps, Some(meth.self_ty.node));
    word(s.s, ~" ");
    print_block_with_attrs(s, meth.body, meth.attrs);
}

fn print_outer_attributes(s: ps, attrs: ~[ast::attribute]) {
    let mut count = 0;
    for attrs.each |attr| {
        match attr.node.style {
          ast::attr_outer => { print_attribute(s, attr); count += 1; }
          _ => {/* fallthrough */ }
        }
    }
    if count > 0 { hardbreak_if_not_bol(s); }
}

fn print_inner_attributes(s: ps, attrs: ~[ast::attribute]) {
    let mut count = 0;
    for attrs.each |attr| {
        match attr.node.style {
          ast::attr_inner => {
            print_attribute(s, attr);
            if !attr.node.is_sugared_doc {
                word(s.s, ~";");
            }
            count += 1;
          }
          _ => {/* fallthrough */ }
        }
    }
    if count > 0 { hardbreak_if_not_bol(s); }
}

fn print_attribute(s: ps, attr: ast::attribute) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, attr.span.lo);
    if attr.node.is_sugared_doc {
        let meta = attr::attr_meta(attr);
        let comment = attr::get_meta_item_value_str(meta).get();
        word(s.s, comment);
    } else {
        word(s.s, ~"#[");
        print_meta_item(s, @attr.node.value);
        word(s.s, ~"]");
    }
}


fn print_stmt(s: ps, st: ast::stmt) {
    maybe_print_comment(s, st.span.lo);
    match st.node {
      ast::stmt_decl(decl, _) => {
        print_decl(s, decl);
      }
      ast::stmt_expr(expr, _) => {
        space_if_not_bol(s);
        print_expr(s, expr);
      }
      ast::stmt_semi(expr, _) => {
        space_if_not_bol(s);
        print_expr(s, expr);
        word(s.s, ~";");
      }
    }
    if parse::classify::stmt_ends_with_semi(st) { word(s.s, ~";"); }
    maybe_print_trailing_comment(s, st.span, None::<uint>);
}

fn print_block(s: ps, blk: ast::blk) {
    print_possibly_embedded_block(s, blk, block_normal, indent_unit);
}

fn print_block_unclosed(s: ps, blk: ast::blk) {
    print_possibly_embedded_block_(s, blk, block_normal, indent_unit, ~[],
                                 false);
}

fn print_block_unclosed_indent(s: ps, blk: ast::blk, indented: uint) {
    print_possibly_embedded_block_(s, blk, block_normal, indented, ~[],
                                   false);
}

fn print_block_with_attrs(s: ps, blk: ast::blk, attrs: ~[ast::attribute]) {
    print_possibly_embedded_block_(s, blk, block_normal, indent_unit, attrs,
                                  true);
}

enum embed_type { block_block_fn, block_normal, }

fn print_possibly_embedded_block(s: ps, blk: ast::blk, embedded: embed_type,
                                 indented: uint) {
    print_possibly_embedded_block_(
        s, blk, embedded, indented, ~[], true);
}

fn print_possibly_embedded_block_(s: ps, blk: ast::blk, embedded: embed_type,
                                  indented: uint, attrs: ~[ast::attribute],
                                  close_box: bool) {
    match blk.node.rules {
      ast::unchecked_blk => word(s.s, ~"unchecked"),
      ast::unsafe_blk => word(s.s, ~"unsafe"),
      ast::default_blk => ()
    }
    maybe_print_comment(s, blk.span.lo);
    let ann_node = node_block(s, blk);
    s.ann.pre(ann_node);
    match embedded {
      block_block_fn => end(s),
      block_normal => bopen(s)
    }

    print_inner_attributes(s, attrs);

    for blk.node.view_items.each |vi| { print_view_item(s, vi); }
    for blk.node.stmts.each |st| {
        print_stmt(s, *st);
    }
    match blk.node.expr {
      Some(expr) => {
        space_if_not_bol(s);
        print_expr(s, expr);
        maybe_print_trailing_comment(s, expr.span, Some(blk.span.hi));
      }
      _ => ()
    }
    bclose_maybe_open(s, blk.span, indented, close_box);
    s.ann.post(ann_node);
}

// return and fail, without arguments cannot appear is the discriminant of if,
// alt, do, & while unambiguously without being parenthesized
fn print_maybe_parens_discrim(s: ps, e: @ast::expr) {
    let disambig = match e.node {
      ast::expr_ret(None)
      | ast::expr_fail(None)
      | ast::expr_again(*) => true,
      _ => false
    };
    if disambig { popen(s); }
    print_expr(s, e);
    if disambig { pclose(s); }
}

fn print_if(s: ps, test: @ast::expr, blk: ast::blk,
            elseopt: Option<@ast::expr>, chk: bool) {
    head(s, ~"if");
    if chk { word_nbsp(s, ~"check"); }
    print_maybe_parens_discrim(s, test);
    space(s.s);
    print_block(s, blk);
    fn do_else(s: ps, els: Option<@ast::expr>) {
        match els {
          Some(_else) => {
            match _else.node {
              // "another else-if"
              ast::expr_if(i, t, e) => {
                cbox(s, indent_unit - 1u);
                ibox(s, 0u);
                word(s.s, ~" else if ");
                print_maybe_parens_discrim(s, i);
                space(s.s);
                print_block(s, t);
                do_else(s, e);
              }
              // "final else"
              ast::expr_block(b) => {
                cbox(s, indent_unit - 1u);
                ibox(s, 0u);
                word(s.s, ~" else ");
                print_block(s, b);
              }
              // BLEAH, constraints would be great here
              _ => {
                  fail ~"print_if saw if with weird alternative";
              }
            }
          }
          _ => {/* fall through */ }
        }
    }
    do_else(s, elseopt);
}

fn print_mac(s: ps, m: ast::mac) {
    match m.node {
      ast::mac_invoc(path, arg, _body) => {
        word(s.s, ~"#");
        print_path(s, path, false);
        match arg {
          Some(@{node: ast::expr_vec(_, _), _}) => (),
          _ => word(s.s, ~" ")
        }
        option::iter(arg, |a| print_expr(s, a));
        // FIXME: extension 'body' (#2339)
      }
      ast::mac_invoc_tt(pth, tts) => {
        print_path(s, pth, false);
        word(s.s, ~"!");
        popen(s);
        for tts.each() |tt| { print_tt(s, tt); }
        pclose(s);
      }
      ast::mac_ellipsis => word(s.s, ~"..."),
      ast::mac_var(v) => word(s.s, fmt!("$%u", v)),
      _ => { /* fixme */ }
    }
}

fn print_vstore(s: ps, t: ast::vstore) {
    match t {
      ast::vstore_fixed(Some(i)) => word(s.s, fmt!("%u", i)),
      ast::vstore_fixed(None) => word(s.s, ~"_"),
      ast::vstore_uniq => word(s.s, ~"~"),
      ast::vstore_box => word(s.s, ~"@"),
      ast::vstore_slice(r) => match r.node {
        ast::re_anon => word(s.s, ~"&"),
        ast::re_named(name) => {
            word(s.s, ~"&");
            print_ident(s, name);
            word(s.s, ~".");
        }
      }
    }
}

fn print_expr(s: ps, &&expr: @ast::expr) {
    fn print_field(s: ps, field: ast::field) {
        ibox(s, indent_unit);
        if field.node.mutbl == ast::m_mutbl { word_nbsp(s, ~"mut"); }
        print_ident(s, field.node.ident);
        word_space(s, ~":");
        print_expr(s, field.node.expr);
        end(s);
    }
    fn get_span(field: ast::field) -> codemap::span { return field.span; }

    maybe_print_comment(s, expr.span.lo);
    ibox(s, indent_unit);
    let ann_node = node_expr(s, expr);
    s.ann.pre(ann_node);
    match expr.node {
      ast::expr_vstore(e, v) => match v {
        ast::vstore_fixed(_) => {
            print_expr(s, e);
              word(s.s, ~"/");
              print_vstore(s, v);
          }
        _ => {
            print_vstore(s, v);
              print_expr(s, e);
          }
      },
      ast::expr_vec(exprs, mutbl) => {
        ibox(s, indent_unit);
        word(s.s, ~"[");
        if mutbl == ast::m_mutbl {
            word(s.s, ~"mut");
            if vec::len(exprs) > 0u { nbsp(s); }
        }
        commasep_exprs(s, inconsistent, exprs);
        word(s.s, ~"]");
        end(s);
      }

      ast::expr_repeat(element, count, mutbl) => {
        ibox(s, indent_unit);
        word(s.s, ~"[");
        if mutbl == ast::m_mutbl {
            word(s.s, ~"mut");
            nbsp(s);
        }
        print_expr(s, element);
        word(s.s, ~",");
        word(s.s, ~"..");
        print_expr(s, count);
        word(s.s, ~"]");
        end(s);
      }

      ast::expr_rec(fields, wth) => {
        word(s.s, ~"{");
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        match wth {
          Some(expr) => {
            ibox(s, indent_unit);
            word(s.s, ~",");
            space(s.s);
            word(s.s, ~"..");
            print_expr(s, expr);
            end(s);
          }
          _ => word(s.s, ~",")
        }
        word(s.s, ~"}");
      }
      ast::expr_struct(path, fields, wth) => {
        print_path(s, path, true);
        word(s.s, ~"{");
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        match wth {
            Some(expr) => {
                if vec::len(fields) > 0u { space(s.s); }
                ibox(s, indent_unit);
                word(s.s, ~",");
                space(s.s);
                word(s.s, ~"..");
                print_expr(s, expr);
                end(s);
            }
            _ => word(s.s, ~",")
        }
        word(s.s, ~"}");
      }
      ast::expr_tup(exprs) => {
        popen(s);
        commasep_exprs(s, inconsistent, exprs);
        pclose(s);
      }
      ast::expr_call(func, args, has_block) => {
        let mut base_args = args;
        let blk = if has_block {
            let blk_arg = vec::pop(base_args);
            match blk_arg.node {
              ast::expr_loop_body(_) => { head(s, ~"for"); }
              ast::expr_do_body(_) => { head(s, ~"do"); }
              _ => {}
            }
            Some(blk_arg)
        } else { None };
        print_expr_parens_if_not_bot(s, func);
        if !has_block || vec::len(base_args) > 0u {
            popen(s);
            commasep_exprs(s, inconsistent, base_args);
            pclose(s);
        }
        if has_block {
            nbsp(s);
            match blk.get().node {
              // need to handle closures specifically
              ast::expr_do_body(e) | ast::expr_loop_body(e) => {
                end(s); // we close our head box; closure
                        // will create it's own.
                print_expr(s, e);
                end(s); // close outer box, as closures don't
              }
              _ => {
                // not sure if this can happen.
                print_expr(s, blk.get());
              }
            }
        }
      }
      ast::expr_binary(op, lhs, rhs) => {
        let prec = operator_prec(op);
        print_op_maybe_parens(s, lhs, prec);
        space(s.s);
        word_space(s, ast_util::binop_to_str(op));
        print_op_maybe_parens(s, rhs, prec + 1u);
      }
      ast::expr_unary(op, expr) => {
        word(s.s, ast_util::unop_to_str(op));
        print_op_maybe_parens(s, expr, parse::prec::unop_prec);
      }
      ast::expr_addr_of(m, expr) => {
        word(s.s, ~"&");
        print_mutability(s, m);
        print_expr(s, expr);
      }
      ast::expr_lit(lit) => print_literal(s, lit),
      ast::expr_cast(expr, ty) => {
        print_op_maybe_parens(s, expr, parse::prec::as_prec);
        space(s.s);
        word_space(s, ~"as");
        print_type_ex(s, ty, true);
      }
      ast::expr_if(test, blk, elseopt) => {
        print_if(s, test, blk, elseopt, false);
      }
      ast::expr_while(test, blk) => {
        head(s, ~"while");
        print_maybe_parens_discrim(s, test);
        space(s.s);
        print_block(s, blk);
      }
      ast::expr_loop(blk, opt_ident) => {
        head(s, ~"loop");
        space(s.s);
        option::iter(opt_ident, |ident| {print_ident(s, ident); space(s.s)});
        print_block(s, blk);
      }
      ast::expr_match(expr, arms) => {
        cbox(s, alt_indent_unit);
        ibox(s, 4u);
        word_nbsp(s, ~"match");
        print_maybe_parens_discrim(s, expr);
        space(s.s);
        bopen(s);
        let len = arms.len();
        for arms.eachi |i, arm| {
            space(s.s);
            cbox(s, alt_indent_unit);
            ibox(s, 0u);
            let mut first = true;
            for arm.pats.each |p| {
                if first {
                    first = false;
                } else { space(s.s); word_space(s, ~"|"); }
                print_pat(s, p);
            }
            space(s.s);
            match arm.guard {
              Some(e) => {
                word_space(s, ~"if");
                print_expr(s, e);
                space(s.s);
              }
              None => ()
            }
            word_space(s, ~"=>");

            // Extract the expression from the extra block the parser adds
            // in the case of foo => expr
            if arm.body.node.view_items.is_empty() &&
                arm.body.node.stmts.is_empty() &&
                arm.body.node.rules == ast::default_blk &&
                arm.body.node.expr.is_some()
            {
                match arm.body.node.expr {
                    Some(expr) => {
                        match expr.node {
                            ast::expr_block(blk) => {
                                // the block will close the pattern's ibox
                                print_block_unclosed_indent(
                                    s, blk, alt_indent_unit);
                            }
                            _ => {
                                end(s); // close the ibox for the pattern
                                print_expr(s, expr);
                            }
                        }
                        if !expr_is_simple_block(expr)
                            && i < len - 1 {
                            word(s.s, ~",");
                        }
                        end(s); // close enclosing cbox
                    }
                    None => fail
                }
            } else {
                // the block will close the pattern's ibox
                print_block_unclosed_indent(s, arm.body, alt_indent_unit);
            }
        }
        bclose_(s, expr.span, alt_indent_unit);
      }
      ast::expr_fn(proto, decl, body, cap_clause) => {
        // containing cbox, will be closed by print-block at }
        cbox(s, indent_unit);
        // head-box, will be closed by print-block at start
        ibox(s, 0u);
        word(s.s, fn_header_info_to_str(None, None, Some(proto)));
        print_fn_args_and_ret(s, decl, *cap_clause, None);
        space(s.s);
        print_block(s, body);
      }
      ast::expr_fn_block(decl, body, cap_clause) => {
        // in do/for blocks we don't want to show an empty
        // argument list, but at this point we don't know which
        // we are inside.
        //
        // if !decl.inputs.is_empty() {
        print_fn_block_args(s, decl, *cap_clause);
        space(s.s);
        // }
        assert body.node.stmts.is_empty();
        assert body.node.expr.is_some();
        // we extract the block, so as not to create another set of boxes
        match body.node.expr.get().node {
            ast::expr_block(blk) => {
                print_block_unclosed(s, blk);
            }
            _ => {
                // this is a bare expression
                print_expr(s, body.node.expr.get());
                end(s); // need to close a box
            }
        }
        // a box will be closed by print_expr, but we didn't want an overall
        // wrapper so we closed the corresponding opening. so create an
        // empty box to satisfy the close.
        ibox(s, 0);
      }
      ast::expr_loop_body(body) => {
        print_expr(s, body);
      }
      ast::expr_do_body(body) => {
        print_expr(s, body);
      }
      ast::expr_block(blk) => {
        // containing cbox, will be closed by print-block at }
        cbox(s, indent_unit);
        // head-box, will be closed by print-block after {
        ibox(s, 0u);
        print_block(s, blk);
      }
      ast::expr_copy(e) => { word_space(s, ~"copy"); print_expr(s, e); }
      ast::expr_unary_move(e) => {
          popen(s);
          word_space(s, ~"move");
          print_expr(s, e);
          pclose(s);
      }
      ast::expr_move(lhs, rhs) => {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, ~"<-");
        print_expr(s, rhs);
      }
      ast::expr_assign(lhs, rhs) => {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, ~"=");
        print_expr(s, rhs);
      }
      ast::expr_swap(lhs, rhs) => {
        print_expr(s, lhs);
        space(s.s);
        word_space(s, ~"<->");
        print_expr(s, rhs);
      }
      ast::expr_assign_op(op, lhs, rhs) => {
        print_expr(s, lhs);
        space(s.s);
        word(s.s, ast_util::binop_to_str(op));
        word_space(s, ~"=");
        print_expr(s, rhs);
      }
      ast::expr_field(expr, id, tys) => {
        // Deal with '10.x'
        if ends_in_lit_int(expr) {
            popen(s); print_expr(s, expr); pclose(s);
        } else {
            print_expr_parens_if_not_bot(s, expr);
        }
        word(s.s, ~".");
        print_ident(s, id);
        if vec::len(tys) > 0u {
            word(s.s, ~"::<");
            commasep(s, inconsistent, tys, print_type);
            word(s.s, ~">");
        }
      }
      ast::expr_index(expr, index) => {
        print_expr_parens_if_not_bot(s, expr);
        word(s.s, ~"[");
        print_expr(s, index);
        word(s.s, ~"]");
      }
      ast::expr_path(path) => print_path(s, path, true),
      ast::expr_fail(maybe_fail_val) => {
        word(s.s, ~"fail");
        match maybe_fail_val {
          Some(expr) => { word(s.s, ~" "); print_expr(s, expr); }
          _ => ()
        }
      }
      ast::expr_break(opt_ident) => {
        word(s.s, ~"break");
        space(s.s);
        option::iter(opt_ident, |ident| {print_ident(s, ident); space(s.s)});
      }
      ast::expr_again(opt_ident) => {
        word(s.s, ~"loop");
        space(s.s);
        option::iter(opt_ident, |ident| {print_ident(s, ident); space(s.s)});
      }
      ast::expr_ret(result) => {
        word(s.s, ~"return");
        match result {
          Some(expr) => { word(s.s, ~" "); print_expr(s, expr); }
          _ => ()
        }
      }
      ast::expr_log(lvl, lexp, expr) => {
        match lvl {
          ast::debug => { word_nbsp(s, ~"log"); print_expr(s, expr); }
          ast::error => { word_nbsp(s, ~"log_err"); print_expr(s, expr); }
          ast::other => {
            word_nbsp(s, ~"log");
            popen(s);
            print_expr(s, lexp);
            word(s.s, ~",");
            space_if_not_bol(s);
            print_expr(s, expr);
            pclose(s);
          }
        }
      }
      ast::expr_assert(expr) => {
        word_nbsp(s, ~"assert");
        print_expr(s, expr);
      }
      ast::expr_mac(m) => print_mac(s, m),
    }
    s.ann.post(ann_node);
    end(s);
}

fn print_expr_parens_if_not_bot(s: ps, ex: @ast::expr) {
    let parens = match ex.node {
      ast::expr_fail(_) | ast::expr_ret(_) |
      ast::expr_binary(_, _, _) | ast::expr_unary(_, _) |
      ast::expr_move(_, _) | ast::expr_copy(_) |
      ast::expr_assign(_, _) |
      ast::expr_assign_op(_, _, _) | ast::expr_swap(_, _) |
      ast::expr_log(_, _, _) | ast::expr_assert(_) |
      ast::expr_call(_, _, true) |
      ast::expr_vstore(_, _) => true,
      _ => false
    };
    if parens { popen(s); }
    print_expr(s, ex);
    if parens { pclose(s); }
}

fn print_local_decl(s: ps, loc: @ast::local) {
    print_pat(s, loc.node.pat);
    match loc.node.ty.node {
      ast::ty_infer => (),
      _ => { word_space(s, ~":"); print_type(s, loc.node.ty); }
    }
}

fn print_decl(s: ps, decl: @ast::decl) {
    maybe_print_comment(s, decl.span.lo);
    match decl.node {
      ast::decl_local(locs) => {
        space_if_not_bol(s);
        ibox(s, indent_unit);
        word_nbsp(s, ~"let");

        // if any are mut, all are mut
        if vec::any(locs, |l| l.node.is_mutbl) {
            assert vec::all(locs, |l| l.node.is_mutbl);
            word_nbsp(s, ~"mut");
        }

        fn print_local(s: ps, &&loc: @ast::local) {
            ibox(s, indent_unit);
            print_local_decl(s, loc);
            end(s);
            match loc.node.init {
              Some(init) => {
                nbsp(s);
                match init.op {
                  ast::init_assign => word_space(s, ~"="),
                  ast::init_move => word_space(s, ~"<-")
                }
                print_expr(s, init.expr);
              }
              _ => ()
            }
        }
        commasep(s, consistent, locs, print_local);
        end(s);
      }
      ast::decl_item(item) => print_item(s, item)
    }
}

fn print_ident(s: ps, ident: ast::ident) { word(s.s, *s.intr.get(ident)); }

fn print_for_decl(s: ps, loc: @ast::local, coll: @ast::expr) {
    print_local_decl(s, loc);
    space(s.s);
    word_space(s, ~"in");
    print_expr(s, coll);
}

fn print_path(s: ps, &&path: @ast::path, colons_before_params: bool) {
    maybe_print_comment(s, path.span.lo);
    if path.global { word(s.s, ~"::"); }
    let mut first = true;
    for path.idents.each |id| {
        if first { first = false; } else { word(s.s, ~"::"); }
        print_ident(s, id);
    }
    if path.rp.is_some() || !path.types.is_empty() {
        if colons_before_params { word(s.s, ~"::"); }

        match path.rp {
          None => { /* ok */ }
          Some(r) => {
            word(s.s, ~"/");
            print_region(s, r);
          }
        }

        if !path.types.is_empty() {
            word(s.s, ~"<");
            commasep(s, inconsistent, path.types, print_type);
            word(s.s, ~">");
        }
    }
}

fn print_pat(s: ps, &&pat: @ast::pat) {
    maybe_print_comment(s, pat.span.lo);
    let ann_node = node_pat(s, pat);
    s.ann.pre(ann_node);
    /* Pat isn't normalized, but the beauty of it
     is that it doesn't matter */
    match pat.node {
      ast::pat_wild => word(s.s, ~"_"),
      ast::pat_ident(binding_mode, path, sub) => {
        match binding_mode {
          ast::bind_by_ref(mutbl) => {
            word_nbsp(s, ~"ref");
            print_mutability(s, mutbl);
          }
          ast::bind_by_move => {
            word_nbsp(s, ~"move");
          }
          ast::bind_by_implicit_ref |
          ast::bind_by_value => {}
        }
        print_path(s, path, true);
        match sub {
          Some(p) => { word(s.s, ~"@"); print_pat(s, p); }
          None => ()
        }
      }
      ast::pat_enum(path, args_) => {
        print_path(s, path, true);
        match args_ {
          None => word(s.s, ~"(*)"),
          Some(args) => {
            if vec::len(args) > 0u {
              popen(s);
              commasep(s, inconsistent, args, print_pat);
              pclose(s);
            } else { }
          }
        }
      }
      ast::pat_rec(fields, etc) => {
        word(s.s, ~"{");
        fn print_field(s: ps, f: ast::field_pat) {
            cbox(s, indent_unit);
            print_ident(s, f.ident);
            word_space(s, ~":");
            print_pat(s, f.pat);
            end(s);
        }
        fn get_span(f: ast::field_pat) -> codemap::span { return f.pat.span; }
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        if etc {
            if vec::len(fields) != 0u { word_space(s, ~","); }
            word(s.s, ~"_");
        }
        word(s.s, ~"}");
      }
      ast::pat_struct(path, fields, etc) => {
        print_path(s, path, true);
        word(s.s, ~"{");
        fn print_field(s: ps, f: ast::field_pat) {
            cbox(s, indent_unit);
            print_ident(s, f.ident);
            word_space(s, ~":");
            print_pat(s, f.pat);
            end(s);
        }
        fn get_span(f: ast::field_pat) -> codemap::span { return f.pat.span; }
        commasep_cmnt(s, consistent, fields, print_field, get_span);
        if etc {
            if vec::len(fields) != 0u { word_space(s, ~","); }
            word(s.s, ~"_");
        }
        word(s.s, ~"}");
      }
      ast::pat_tup(elts) => {
        popen(s);
        commasep(s, inconsistent, elts, print_pat);
        pclose(s);
      }
      ast::pat_box(inner) => { word(s.s, ~"@"); print_pat(s, inner); }
      ast::pat_uniq(inner) => { word(s.s, ~"~"); print_pat(s, inner); }
      ast::pat_region(inner) => {
          word(s.s, ~"&");
          print_pat(s, inner);
      }
      ast::pat_lit(e) => print_expr(s, e),
      ast::pat_range(begin, end) => {
        print_expr(s, begin);
        space(s.s);
        word(s.s, ~"..");
        print_expr(s, end);
      }
    }
    s.ann.post(ann_node);
}

// Returns whether it printed anything
fn print_self_ty(s: ps, self_ty: ast::self_ty_) -> bool {
    match self_ty {
      ast::sty_static | ast::sty_by_ref => { return false; }
      ast::sty_value => { word(s.s, ~"self"); }
      ast::sty_region(m) => {
        word(s.s, ~"&"); print_mutability(s, m); word(s.s, ~"self");
      }
      ast::sty_box(m) => {
        word(s.s, ~"@"); print_mutability(s, m); word(s.s, ~"self");
      }
      ast::sty_uniq(m) => {
        word(s.s, ~"~"); print_mutability(s, m); word(s.s, ~"self");
      }
    }
    return true;
}

fn print_fn(s: ps, decl: ast::fn_decl, purity: Option<ast::purity>,
            name: ast::ident,
            typarams: ~[ast::ty_param],
            opt_self_ty: Option<ast::self_ty_>) {
    head(s, fn_header_info_to_str(opt_self_ty, purity, None));
    print_ident(s, name);
    print_type_params(s, typarams);
    print_fn_args_and_ret(s, decl, ~[], opt_self_ty);
}

fn print_fn_args(s: ps, decl: ast::fn_decl,
                 cap_items: ~[ast::capture_item],
                 opt_self_ty: Option<ast::self_ty_>) {
    // It is unfortunate to duplicate the commasep logic, but we
    // we want the self type, the args, and the capture clauses all
    // in the same box.
    box(s, 0u, inconsistent);
    let mut first = true;
    for opt_self_ty.each |self_ty| {
        first = !print_self_ty(s, self_ty);
    }

    for decl.inputs.each |arg| {
        if first { first = false; } else { word_space(s, ~","); }
        print_arg(s, arg);
    }

    for cap_items.each |cap_item| {
        if first { first = false; } else { word_space(s, ~","); }
        if cap_item.is_move { word_nbsp(s, ~"move") }
        else { word_nbsp(s, ~"copy") }
        print_ident(s, cap_item.name);
    }

    end(s);
}

fn print_fn_args_and_ret(s: ps, decl: ast::fn_decl,
                         cap_items: ~[ast::capture_item],
                         opt_self_ty: Option<ast::self_ty_>) {
    popen(s);
    print_fn_args(s, decl, cap_items, opt_self_ty);
    pclose(s);

    maybe_print_comment(s, decl.output.span.lo);
    if decl.output.node != ast::ty_nil {
        space_if_not_bol(s);
        word_space(s, ~"->");
        print_type(s, decl.output);
    }
}

fn print_fn_block_args(s: ps, decl: ast::fn_decl,
                       cap_items: ~[ast::capture_item]) {
    word(s.s, ~"|");
    print_fn_args(s, decl, cap_items, None);
    word(s.s, ~"|");
    if decl.output.node != ast::ty_infer {
        space_if_not_bol(s);
        word_space(s, ~"->");
        print_type(s, decl.output);
    }
    maybe_print_comment(s, decl.output.span.lo);
}

fn mode_to_str(m: ast::mode) -> ~str {
    match m {
      ast::expl(ast::by_mutbl_ref) => ~"&",
      ast::expl(ast::by_move) => ~"-",
      ast::expl(ast::by_ref) => ~"&&",
      ast::expl(ast::by_val) => ~"++",
      ast::expl(ast::by_copy) => ~"+",
      ast::infer(_) => ~""
    }
}

fn print_arg_mode(s: ps, m: ast::mode) {
    let ms = mode_to_str(m);
    if ms != ~"" { word(s.s, ms); }
}

fn print_bounds(s: ps, bounds: @~[ast::ty_param_bound]) {
    if vec::len(*bounds) > 0u {
        word(s.s, ~":");
        for vec::each(*bounds) |bound| {
            nbsp(s);
            match bound {
              ast::bound_copy => word(s.s, ~"copy"),
              ast::bound_send => word(s.s, ~"send"),
              ast::bound_const => word(s.s, ~"const"),
              ast::bound_owned => word(s.s, ~"owned"),
              ast::bound_trait(t) => print_type(s, t)
            }
        }
    }
}

fn print_type_params(s: ps, &&params: ~[ast::ty_param]) {
    if vec::len(params) > 0u {
        word(s.s, ~"<");
        fn printParam(s: ps, param: ast::ty_param) {
            print_ident(s, param.ident);
            print_bounds(s, param.bounds);
        }
        commasep(s, inconsistent, params, printParam);
        word(s.s, ~">");
    }
}

fn print_meta_item(s: ps, &&item: @ast::meta_item) {
    ibox(s, indent_unit);
    match item.node {
      ast::meta_word(name) => word(s.s, name),
      ast::meta_name_value(name, value) => {
        word_space(s, name);
        word_space(s, ~"=");
        print_literal(s, @value);
      }
      ast::meta_list(name, items) => {
        word(s.s, name);
        popen(s);
        commasep(s, consistent, items, print_meta_item);
        pclose(s);
      }
    }
    end(s);
}

fn print_view_path(s: ps, &&vp: @ast::view_path) {
    match vp.node {
      ast::view_path_simple(ident, path, namespace, _) => {
        if namespace == ast::module_ns {
            word_space(s, ~"mod");
        }
        if path.idents[vec::len(path.idents)-1u] != ident {
            print_ident(s, ident);
            space(s.s);
            word_space(s, ~"=");
        }
        print_path(s, path, false);
      }

      ast::view_path_glob(path, _) => {
        print_path(s, path, false);
        word(s.s, ~"::*");
      }

      ast::view_path_list(path, idents, _) => {
        print_path(s, path, false);
        word(s.s, ~"::{");
        do commasep(s, inconsistent, idents) |s, w| {
            print_ident(s, w.node.name);
        }
        word(s.s, ~"}");
      }
    }
}

fn print_view_paths(s: ps, vps: ~[@ast::view_path]) {
    commasep(s, inconsistent, vps, print_view_path);
}

fn print_view_item(s: ps, item: @ast::view_item) {
    hardbreak_if_not_bol(s);
    maybe_print_comment(s, item.span.lo);
    print_outer_attributes(s, item.attrs);
    match item.node {
      ast::view_item_use(id, mta, _) => {
        head(s, ~"use");
        print_ident(s, id);
        if vec::len(mta) > 0u {
            popen(s);
            commasep(s, consistent, mta, print_meta_item);
            pclose(s);
        }
      }

      ast::view_item_import(vps) => {
        head(s, ~"import");
        print_view_paths(s, vps);
      }

      ast::view_item_export(vps) => {
        head(s, ~"export");
        print_view_paths(s, vps);
      }
    }
    word(s.s, ~";");
    end(s); // end inner head-block
    end(s); // end outer head-block
}

fn print_op_maybe_parens(s: ps, expr: @ast::expr, outer_prec: uint) {
    let add_them = need_parens(expr, outer_prec);
    if add_them { popen(s); }
    print_expr(s, expr);
    if add_them { pclose(s); }
}

fn print_mutability(s: ps, mutbl: ast::mutability) {
    match mutbl {
      ast::m_mutbl => word_nbsp(s, ~"mut"),
      ast::m_const => word_nbsp(s, ~"const"),
      ast::m_imm => {/* nothing */ }
    }
}

fn print_mt(s: ps, mt: ast::mt) {
    print_mutability(s, mt.mutbl);
    print_type(s, mt.ty);
}

fn print_arg(s: ps, input: ast::arg) {
    ibox(s, indent_unit);
    print_arg_mode(s, input.mode);
    match input.ty.node {
      ast::ty_infer => print_ident(s, input.ident),
      _ => {
        if input.ident != parse::token::special_idents::invalid {
            print_ident(s, input.ident);
            word(s.s, ~":");
            space(s.s);
        }
        print_type(s, input.ty);
      }
    }
    end(s);
}

fn print_ty_fn(s: ps, opt_proto: Option<ast::proto>, purity: ast::purity,
               bounds: @~[ast::ty_param_bound],
               decl: ast::fn_decl, id: Option<ast::ident>,
               tps: Option<~[ast::ty_param]>,
               opt_self_ty: Option<ast::self_ty_>) {
    ibox(s, indent_unit);
    word(s.s, fn_header_info_to_str(opt_self_ty, Some(purity), opt_proto));
    print_bounds(s, bounds);
    match id { Some(id) => { word(s.s, ~" "); print_ident(s, id); } _ => () }
    match tps { Some(tps) => print_type_params(s, tps), _ => () }
    zerobreak(s.s);

    popen(s);
    // It is unfortunate to duplicate the commasep logic, but we
    // we want the self type, the args, and the capture clauses all
    // in the same box.
    box(s, 0u, inconsistent);
    let mut first = true;
    for opt_self_ty.each |self_ty| {
        first = !print_self_ty(s, self_ty);
    }
    for decl.inputs.each |arg| {
        if first { first = false; } else { word_space(s, ~","); }
        print_arg(s, arg);
    }
    end(s);
    pclose(s);

    maybe_print_comment(s, decl.output.span.lo);
    if decl.output.node != ast::ty_nil {
        space_if_not_bol(s);
        ibox(s, indent_unit);
        word_space(s, ~"->");
        if decl.cf == ast::noreturn { word_nbsp(s, ~"!"); }
        else { print_type(s, decl.output); }
        end(s);
    }
    end(s);
}

fn maybe_print_trailing_comment(s: ps, span: codemap::span,
                                next_pos: Option<uint>) {
    let mut cm;
    match s.cm { Some(ccm) => cm = ccm, _ => return }
    match next_comment(s) {
      Some(cmnt) => {
        if cmnt.style != comments::trailing { return; }
        let span_line = codemap::lookup_char_pos(cm, span.hi);
        let comment_line = codemap::lookup_char_pos(cm, cmnt.pos);
        let mut next = cmnt.pos + 1u;
        match next_pos { None => (), Some(p) => next = p }
        if span.hi < cmnt.pos && cmnt.pos < next &&
               span_line.line == comment_line.line {
            print_comment(s, cmnt);
            s.cur_cmnt += 1u;
        }
      }
      _ => ()
    }
}

fn print_remaining_comments(s: ps) {
    // If there aren't any remaining comments, then we need to manually
    // make sure there is a line break at the end.
    if option::is_none(next_comment(s)) { hardbreak(s.s); }
    loop {
        match next_comment(s) {
          Some(cmnt) => { print_comment(s, cmnt); s.cur_cmnt += 1u; }
          _ => break
        }
    }
}

fn print_literal(s: ps, &&lit: @ast::lit) {
    maybe_print_comment(s, lit.span.lo);
    match next_lit(s, lit.span.lo) {
      Some(ltrl) => {
        word(s.s, ltrl.lit);
        return;
      }
      _ => ()
    }
    match lit.node {
      ast::lit_str(st) => print_string(s, *st),
      ast::lit_int(ch, ast::ty_char) => {
        word(s.s, ~"'" + char::escape_default(ch as char) + ~"'");
      }
      ast::lit_int(i, t) => {
        if i < 0_i64 {
            word(s.s,
                 ~"-" + u64::to_str(-i as u64, 10u)
                 + ast_util::int_ty_to_str(t));
        } else {
            word(s.s,
                 u64::to_str(i as u64, 10u)
                 + ast_util::int_ty_to_str(t));
        }
      }
      ast::lit_uint(u, t) => {
        word(s.s,
             u64::to_str(u, 10u)
             + ast_util::uint_ty_to_str(t));
      }
      ast::lit_int_unsuffixed(i) => {
        if i < 0_i64 {
            word(s.s, ~"-" + u64::to_str(-i as u64, 10u));
        } else {
            word(s.s, u64::to_str(i as u64, 10u));
        }
      }
      ast::lit_float(f, t) => {
        word(s.s, *f + ast_util::float_ty_to_str(t));
      }
      ast::lit_nil => word(s.s, ~"()"),
      ast::lit_bool(val) => {
        if val { word(s.s, ~"true"); } else { word(s.s, ~"false"); }
      }
    }
}

fn lit_to_str(l: @ast::lit) -> ~str {
    return to_str(l, print_literal, parse::token::mk_fake_ident_interner());
}

fn next_lit(s: ps, pos: uint) -> Option<comments::lit> {
    match s.literals {
      Some(lits) => {
        while s.cur_lit < vec::len(lits) {
            let ltrl = lits[s.cur_lit];
            if ltrl.pos > pos { return None; }
            s.cur_lit += 1u;
            if ltrl.pos == pos { return Some(ltrl); }
        }
        return None;
      }
      _ => return None
    }
}

fn maybe_print_comment(s: ps, pos: uint) {
    loop {
        match next_comment(s) {
          Some(cmnt) => {
            if cmnt.pos < pos {
                print_comment(s, cmnt);
                s.cur_cmnt += 1u;
            } else { break; }
          }
          _ => break
        }
    }
}

fn print_comment(s: ps, cmnt: comments::cmnt) {
    match cmnt.style {
      comments::mixed => {
        assert (vec::len(cmnt.lines) == 1u);
        zerobreak(s.s);
        word(s.s, cmnt.lines[0]);
        zerobreak(s.s);
      }
      comments::isolated => {
        pprust::hardbreak_if_not_bol(s);
        for cmnt.lines.each |line| {
            // Don't print empty lines because they will end up as trailing
            // whitespace
            if str::is_not_empty(line) { word(s.s, line); }
            hardbreak(s.s);
        }
      }
      comments::trailing => {
        word(s.s, ~" ");
        if vec::len(cmnt.lines) == 1u {
            word(s.s, cmnt.lines[0]);
            hardbreak(s.s);
        } else {
            ibox(s, 0u);
            for cmnt.lines.each |line| {
                if str::is_not_empty(line) { word(s.s, line); }
                hardbreak(s.s);
            }
            end(s);
        }
      }
      comments::blank_line => {
        // We need to do at least one, possibly two hardbreaks.
        let is_semi =
            match s.s.last_token() {
              pp::STRING(s, _) => *s == ~";",
              _ => false
            };
        if is_semi || is_begin(s) || is_end(s) { hardbreak(s.s); }
        hardbreak(s.s);
      }
    }
}

fn print_string(s: ps, st: ~str) {
    word(s.s, ~"\"");
    word(s.s, str::escape_default(st));
    word(s.s, ~"\"");
}

fn to_str<T>(t: T, f: fn@(ps, T), intr: ident_interner) -> ~str {
    let buffer = io::mem_buffer();
    let s = rust_printer(io::mem_buffer_writer(buffer), intr);
    f(s, t);
    eof(s.s);
    io::mem_buffer_str(buffer)
}

fn next_comment(s: ps) -> Option<comments::cmnt> {
    match s.comments {
      Some(cmnts) => {
        if s.cur_cmnt < vec::len(cmnts) {
            return Some(cmnts[s.cur_cmnt]);
        } else { return None::<comments::cmnt>; }
      }
      _ => return None::<comments::cmnt>
    }
}

fn fn_header_info_to_str(opt_sty: Option<ast::self_ty_>,
                         opt_purity: Option<ast::purity>,
                         opt_p: Option<ast::proto>) -> ~str {
    let mut s = match opt_sty {
      Some(ast::sty_static) => ~"static ",
      _ => ~ ""
    };

    match opt_purity {
      Some(ast::impure_fn) => { }
      Some(purity) => {
        str::push_str(s, purity_to_str(purity));
        str::push_char(s, ' ');
      }
      None => {}
    }

    str::push_str(s, opt_proto_to_str(opt_p));

    return s;
}

fn opt_proto_to_str(opt_p: Option<ast::proto>) -> ~str {
    match opt_p {
      None => ~"fn",
      Some(p) => proto_to_str(p)
    }
}

pure fn purity_to_str(p: ast::purity) -> ~str {
    match p {
      ast::impure_fn => ~"impure",
      ast::unsafe_fn => ~"unsafe",
      ast::pure_fn => ~"pure",
      ast::extern_fn => ~"extern"
    }
}

fn print_purity(s: ps, p: ast::purity) {
    match p {
      ast::impure_fn => (),
      _ => word_nbsp(s, purity_to_str(p))
    }
}

fn proto_to_str(p: ast::proto) -> ~str {
    return match p {
      ast::proto_bare => ~"extern fn",
      ast::proto_block => ~"fn&",
      ast::proto_uniq => ~"fn~",
      ast::proto_box => ~"fn@"
    };
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
