// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The Rust abstract syntax tree.

use codemap::{span, FileName};

use core::cast;
use core::cmp;
use core::option::{None, Option, Some};
use core::ptr;
use core::task;
use core::to_bytes;
use core::to_str::ToStr;
use std::serialize::{Encodable, Decodable, Encoder, Decoder};

#[auto_encode]
#[auto_decode]
struct spanned<T> { node: T, span: span }

/* can't import macros yet, so this is copied from token.rs. See its comment
 * there. */
macro_rules! interner_key (
    () => (cast::transmute::<(uint, uint),
            &fn(+v: @@::parse::token::ident_interner)>(
        (-3 as uint, 0u)))
)

// FIXME(#3534): Replace with the struct-based newtype when it's been
// implemented.
struct ident { repr: uint }

impl<S: Encoder> ident: Encodable<S> {
    fn encode(&self, s: &S) {
        let intr = match unsafe {
            task::local_data::local_data_get(interner_key!())
        } {
            None => fail ~"encode: TLS interner not set up",
            Some(intr) => intr
        };

        s.emit_owned_str(*(*intr).get(*self));
    }
}

impl<D: Decoder> ident: Decodable<D> {
    static fn decode(d: &D) -> ident {
        let intr = match unsafe {
            task::local_data::local_data_get(interner_key!())
        } {
            None => fail ~"decode: TLS interner not set up",
            Some(intr) => intr
        };

        (*intr).intern(@d.read_owned_str())
    }
}

impl ident: cmp::Eq {
    pure fn eq(&self, other: &ident) -> bool { (*self).repr == other.repr }
    pure fn ne(&self, other: &ident) -> bool { !(*self).eq(other) }
}

impl ident: to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        self.repr.iter_bytes(lsb0, f)
    }
}

// Functions may or may not have names.
type fn_ident = Option<ident>;

#[auto_encode]
#[auto_decode]
struct path {
    span: span,
    global: bool,
    idents: ~[ident],
    rp: Option<@region>,
    types: ~[@Ty],
}

type crate_num = int;

type node_id = int;

#[auto_encode]
#[auto_decode]
struct def_id {
    crate: crate_num,
    node: node_id,
}

impl def_id : cmp::Eq {
    pure fn eq(&self, other: &def_id) -> bool {
        (*self).crate == (*other).crate && (*self).node == (*other).node
    }
    pure fn ne(&self, other: &def_id) -> bool { !(*self).eq(other) }
}

const local_crate: crate_num = 0;
const crate_node_id: node_id = 0;

#[auto_encode]
#[auto_decode]
// The AST represents all type param bounds as types.
// typeck::collect::compute_bounds matches these against
// the "special" built-in traits (see middle::lang_items) and
// detects Copy, Send, Owned, and Const.
enum ty_param_bound {
    TraitTyParamBound(@Ty),
    RegionTyParamBound
}

#[auto_encode]
#[auto_decode]
struct ty_param {
    ident: ident,
    id: node_id,
    bounds: @~[ty_param_bound]
}

#[auto_encode]
#[auto_decode]
enum def {
    def_fn(def_id, purity),
    def_static_method(/* method */ def_id,
                      /* trait */  Option<def_id>,
                      purity),
    def_self(node_id, bool /* is_implicit */),
    def_self_ty(node_id),
    def_mod(def_id),
    def_foreign_mod(def_id),
    def_const(def_id),
    def_arg(node_id, mode),
    def_local(node_id, bool /* is_mutbl */),
    def_variant(def_id /* enum */, def_id /* variant */),
    def_ty(def_id),
    def_prim_ty(prim_ty),
    def_ty_param(def_id, uint),
    def_binding(node_id, binding_mode),
    def_use(def_id),
    def_upvar(node_id,  // id of closed over var
              @def,     // closed over def
              node_id,  // expr node that creates the closure
              node_id), // id for the block/body of the closure expr
    def_struct(def_id),
    def_typaram_binder(node_id), /* struct, impl or trait with ty params */
    def_region(node_id),
    def_label(node_id)
}

impl def : cmp::Eq {
    pure fn eq(&self, other: &def) -> bool {
        match (*self) {
            def_fn(e0a, e1a) => {
                match (*other) {
                    def_fn(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_static_method(e0a, e1a, e2a) => {
                match (*other) {
                    def_static_method(e0b, e1b, e2b) =>
                    e0a == e0b && e1a == e1b && e2a == e2b,
                    _ => false
                }
            }
            def_self(e0a, e1a) => {
                match (*other) {
                    def_self(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_self_ty(e0a) => {
                match (*other) {
                    def_self_ty(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_mod(e0a) => {
                match (*other) {
                    def_mod(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_foreign_mod(e0a) => {
                match (*other) {
                    def_foreign_mod(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_const(e0a) => {
                match (*other) {
                    def_const(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_arg(e0a, e1a) => {
                match (*other) {
                    def_arg(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_local(e0a, e1a) => {
                match (*other) {
                    def_local(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_variant(e0a, e1a) => {
                match (*other) {
                    def_variant(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_ty(e0a) => {
                match (*other) {
                    def_ty(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_prim_ty(e0a) => {
                match (*other) {
                    def_prim_ty(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_ty_param(e0a, e1a) => {
                match (*other) {
                    def_ty_param(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_binding(e0a, e1a) => {
                match (*other) {
                    def_binding(e0b, e1b) => e0a == e0b && e1a == e1b,
                    _ => false
                }
            }
            def_use(e0a) => {
                match (*other) {
                    def_use(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_upvar(e0a, e1a, e2a, e3a) => {
                match (*other) {
                    def_upvar(e0b, e1b, e2b, e3b) =>
                        e0a == e0b && e1a == e1b && e2a == e2b && e3a == e3b,
                    _ => false
                }
            }
            def_struct(e0a) => {
                match (*other) {
                    def_struct(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_typaram_binder(e0a) => {
                match (*other) {
                    def_typaram_binder(e1a) => e0a == e1a,
                    _ => false
                }
            }
            def_region(e0a) => {
                match (*other) {
                    def_region(e0b) => e0a == e0b,
                    _ => false
                }
            }
            def_label(e0a) => {
                match (*other) {
                    def_label(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &def) -> bool { !(*self).eq(other) }
}

// The set of meta_items that define the compilation environment of the crate,
// used to drive conditional compilation
type crate_cfg = ~[@meta_item];

type crate = spanned<crate_>;

struct crate_ {
    module: _mod,
    attrs: ~[attribute],
    config: crate_cfg,
}

type meta_item = spanned<meta_item_>;

#[auto_encode]
#[auto_decode]
enum meta_item_ {
    meta_word(~str),
    meta_list(~str, ~[@meta_item]),
    meta_name_value(~str, lit),
}

type blk = spanned<blk_>;

#[auto_encode]
#[auto_decode]
struct blk_ {
    view_items: ~[@view_item],
    stmts: ~[@stmt],
    expr: Option<@expr>,
    id: node_id,
    rules: blk_check_mode,
}

#[auto_encode]
#[auto_decode]
struct pat {
    id: node_id,
    node: pat_,
    span: span,
}

#[auto_encode]
#[auto_decode]
struct field_pat {
    ident: ident,
    pat: @pat,
}

#[auto_encode]
#[auto_decode]
enum binding_mode {
    bind_by_value,
    bind_by_move,
    bind_by_ref(mutability),
    bind_infer
}

impl binding_mode : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          bind_by_value => 0u8.iter_bytes(lsb0, f),

          bind_by_move => 1u8.iter_bytes(lsb0, f),

          bind_by_ref(ref m) =>
          to_bytes::iter_bytes_2(&2u8, m, lsb0, f),

          bind_infer =>
          3u8.iter_bytes(lsb0, f),
        }
    }
}

impl binding_mode : cmp::Eq {
    pure fn eq(&self, other: &binding_mode) -> bool {
        match (*self) {
            bind_by_value => {
                match (*other) {
                    bind_by_value => true,
                    _ => false
                }
            }
            bind_by_move => {
                match (*other) {
                    bind_by_move => true,
                    _ => false
                }
            }
            bind_by_ref(e0a) => {
                match (*other) {
                    bind_by_ref(e0b) => e0a == e0b,
                    _ => false
                }
            }
            bind_infer => {
                match (*other) {
                    bind_infer => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &binding_mode) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum pat_ {
    pat_wild,
    // A pat_ident may either be a new bound variable,
    // or a nullary enum (in which case the second field
    // is None).
    // In the nullary enum case, the parser can't determine
    // which it is. The resolver determines this, and
    // records this pattern's node_id in an auxiliary
    // set (of "pat_idents that refer to nullary enums")
    pat_ident(binding_mode, @path, Option<@pat>),
    pat_enum(@path, Option<~[@pat]>), // "none" means a * pattern where
                                  // we don't bind the fields to names
    pat_rec(~[field_pat], bool),
    pat_struct(@path, ~[field_pat], bool),
    pat_tup(~[@pat]),
    pat_box(@pat),
    pat_uniq(@pat),
    pat_region(@pat), // borrowed pointer pattern
    pat_lit(@expr),
    pat_range(@expr, @expr),
    pat_vec(~[@pat], Option<@pat>)
}

#[auto_encode]
#[auto_decode]
enum mutability { m_mutbl, m_imm, m_const, }

impl mutability : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl mutability : cmp::Eq {
    pure fn eq(&self, other: &mutability) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &mutability) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
#[deriving_eq]
pub enum Proto {
    ProtoBare,     // bare functions (deprecated)
    ProtoUniq,     // ~fn
    ProtoBox,      // @fn
    ProtoBorrowed, // &fn
}

impl Proto : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as uint).iter_bytes(lsb0, f);
    }
}

#[auto_encode]
#[auto_decode]
enum vstore {
    // FIXME (#3469): Change uint to @expr (actually only constant exprs)
    vstore_fixed(Option<uint>),   // [1,2,3,4]
    vstore_uniq,                  // ~[1,2,3,4]
    vstore_box,                   // @[1,2,3,4]
    vstore_slice(@region)         // &[1,2,3,4](foo)?
}

#[auto_encode]
#[auto_decode]
enum expr_vstore {
    // FIXME (#3469): Change uint to @expr (actually only constant exprs)
    expr_vstore_fixed(Option<uint>),   // [1,2,3,4]
    expr_vstore_uniq,                  // ~[1,2,3,4]
    expr_vstore_box,                   // @[1,2,3,4]
    expr_vstore_mut_box,               // @mut [1,2,3,4]
    expr_vstore_slice,                 // &[1,2,3,4]
    expr_vstore_mut_slice,             // &mut [1,2,3,4]
}

pure fn is_blockish(p: Proto) -> bool {
    match p {
        ProtoBorrowed => true,
        ProtoBare | ProtoUniq | ProtoBox => false
    }
}

#[auto_encode]
#[auto_decode]
enum binop {
    add,
    subtract,
    mul,
    div,
    rem,
    and,
    or,
    bitxor,
    bitand,
    bitor,
    shl,
    shr,
    eq,
    lt,
    le,
    ne,
    ge,
    gt,
}

impl binop : cmp::Eq {
    pure fn eq(&self, other: &binop) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &binop) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum unop {
    box(mutability),
    uniq(mutability),
    deref,
    not,
    neg
}

impl unop : cmp::Eq {
    pure fn eq(&self, other: &unop) -> bool {
        match (*self) {
            box(e0a) => {
                match (*other) {
                    box(e0b) => e0a == e0b,
                    _ => false
                }
            }
            uniq(e0a) => {
                match (*other) {
                    uniq(e0b) => e0a == e0b,
                    _ => false
                }
            }
            deref => {
                match (*other) {
                    deref => true,
                    _ => false
                }
            }
            not => {
                match (*other) {
                    not => true,
                    _ => false
                }
            }
            neg => {
                match (*other) {
                    neg => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &unop) -> bool {
        !(*self).eq(other)
    }
}

// Generally, after typeck you can get the inferred value
// using ty::resolved_T(...).
#[auto_encode]
#[auto_decode]
enum inferable<T> {
    expl(T),
    infer(node_id)
}

impl<T: to_bytes::IterBytes> inferable<T> : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        match *self {
          expl(ref t) =>
          to_bytes::iter_bytes_2(&0u8, t, lsb0, f),

          infer(ref n) =>
          to_bytes::iter_bytes_2(&1u8, n, lsb0, f),
        }
    }
}

impl<T:cmp::Eq> inferable<T> : cmp::Eq {
    pure fn eq(&self, other: &inferable<T>) -> bool {
        match (*self) {
            expl(ref e0a) => {
                match (*other) {
                    expl(ref e0b) => (*e0a) == (*e0b),
                    _ => false
                }
            }
            infer(e0a) => {
                match (*other) {
                    infer(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &inferable<T>) -> bool { !(*self).eq(other) }
}

// "resolved" mode: the real modes.
#[auto_encode]
#[auto_decode]
enum rmode { by_ref, by_val, by_move, by_copy }

impl rmode : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}


impl rmode : cmp::Eq {
    pure fn eq(&self, other: &rmode) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &rmode) -> bool { !(*self).eq(other) }
}

// inferable mode.
type mode = inferable<rmode>;

type stmt = spanned<stmt_>;

#[auto_encode]
#[auto_decode]
enum stmt_ {
    stmt_decl(@decl, node_id),

    // expr without trailing semi-colon (must have unit type):
    stmt_expr(@expr, node_id),

    // expr with trailing semi-colon (may have any type):
    stmt_semi(@expr, node_id),

    // bool: is there a trailing sem-colon?
    stmt_mac(mac, bool),
}

// FIXME (pending discussion of #1697, #2178...): local should really be
// a refinement on pat.
#[auto_encode]
#[auto_decode]
struct local_ {
    is_mutbl: bool,
    ty: @Ty,
    pat: @pat,
    init: Option<@expr>,
    id: node_id,
}

type local = spanned<local_>;

type decl = spanned<decl_>;

#[auto_encode]
#[auto_decode]
enum decl_ { decl_local(~[@local]), decl_item(@item), }

#[auto_encode]
#[auto_decode]
struct arm {
    pats: ~[@pat],
    guard: Option<@expr>,
    body: blk,
}

#[auto_encode]
#[auto_decode]
struct field_ {
    mutbl: mutability,
    ident: ident,
    expr: @expr,
}

type field = spanned<field_>;

#[auto_encode]
#[auto_decode]
enum blk_check_mode { default_blk, unsafe_blk, }

impl blk_check_mode : cmp::Eq {
    pure fn eq(&self, other: &blk_check_mode) -> bool {
        match ((*self), (*other)) {
            (default_blk, default_blk) => true,
            (unsafe_blk, unsafe_blk) => true,
            (default_blk, _) => false,
            (unsafe_blk, _) => false,
        }
    }
    pure fn ne(&self, other: &blk_check_mode) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
struct expr {
    id: node_id,
    // Extra node ID is only used for index, assign_op, unary, binary, method
    // call
    callee_id: node_id,
    node: expr_,
    span: span,
}

#[auto_encode]
#[auto_decode]
enum log_level { error, debug, log_other }
// 0 = error, 1 = debug, 2 = log_other

#[auto_encode]
#[auto_decode]
enum expr_ {
    expr_vstore(@expr, expr_vstore),
    expr_vec(~[@expr], mutability),
    expr_rec(~[field], Option<@expr>),
    expr_call(@expr, ~[@expr], bool), // True iff last argument is a block
    expr_method_call(@expr, ident, ~[@Ty], ~[@expr], bool), // Ditto
    expr_tup(~[@expr]),
    expr_binary(binop, @expr, @expr),
    expr_unary(unop, @expr),
    expr_lit(@lit),
    expr_cast(@expr, @Ty),
    expr_if(@expr, blk, Option<@expr>),
    expr_while(@expr, blk),
    /* Conditionless loop (can be exited with break, cont, ret, or fail)
       Same semantics as while(true) { body }, but typestate knows that the
       (implicit) condition is always true. */
    expr_loop(blk, Option<ident>),
    expr_match(@expr, ~[arm]),
    expr_fn(Proto, fn_decl, blk, capture_clause),
    expr_fn_block(fn_decl, blk, capture_clause),
    // Inner expr is always an expr_fn_block. We need the wrapping node to
    // easily type this (a function returning nil on the inside but bool on
    // the outside).
    expr_loop_body(@expr),
    // Like expr_loop_body but for 'do' blocks
    expr_do_body(@expr),
    expr_block(blk),

    expr_copy(@expr),
    expr_unary_move(@expr),
    expr_assign(@expr, @expr),
    expr_swap(@expr, @expr),
    expr_assign_op(binop, @expr, @expr),
    expr_field(@expr, ident, ~[@Ty]),
    expr_index(@expr, @expr),
    expr_path(@path),
    expr_addr_of(mutability, @expr),
    expr_fail(Option<@expr>),
    expr_break(Option<ident>),
    expr_again(Option<ident>),
    expr_ret(Option<@expr>),
    expr_log(log_level, @expr, @expr),

    /* just an assert */
    expr_assert(@expr),

    expr_mac(mac),

    // A struct literal expression.
    expr_struct(@path, ~[field], Option<@expr>),

    // A vector literal constructed from one repeated element.
    expr_repeat(@expr /* element */, @expr /* count */, mutability),

    // No-op: used solely so we can pretty-print faithfully
    expr_paren(@expr)
}

#[auto_encode]
#[auto_decode]
struct capture_item_ {
    id: int,
    is_move: bool,
    name: ident, // Currently, can only capture a local var.
    span: span,
}

type capture_item = @capture_item_;

type capture_clause = @~[capture_item];

//
// When the main rust parser encounters a syntax-extension invocation, it
// parses the arguments to the invocation as a token-tree. This is a very
// loose structure, such that all sorts of different AST-fragments can
// be passed to syntax extensions using a uniform type.
//
// If the syntax extension is an MBE macro, it will attempt to match its
// LHS "matchers" against the provided token tree, and if it finds a
// match, will transcribe the RHS token tree, splicing in any captured
// macro_parser::matched_nonterminals into the tt_nonterminals it finds.
//
// The RHS of an MBE macro is the only place a tt_nonterminal or tt_seq
// makes any real sense. You could write them elsewhere but nothing
// else knows what to do with them, so you'll probably get a syntax
// error.
//
#[auto_encode]
#[auto_decode]
#[doc="For macro invocations; parsing is delegated to the macro"]
enum token_tree {
    tt_tok(span, ::parse::token::Token),
    tt_delim(~[token_tree]),
    // These only make sense for right-hand-sides of MBE macros
    tt_seq(span, ~[token_tree], Option<::parse::token::Token>, bool),
    tt_nonterminal(span, ident)
}

//
// Matchers are nodes defined-by and recognized-by the main rust parser and
// language, but they're only ever found inside syntax-extension invocations;
// indeed, the only thing that ever _activates_ the rules in the rust parser
// for parsing a matcher is a matcher looking for the 'matchers' nonterminal
// itself. Matchers represent a small sub-language for pattern-matching
// token-trees, and are thus primarily used by the macro-defining extension
// itself.
//
// match_tok
// ---------
//
//     A matcher that matches a single token, denoted by the token itself. So
//     long as there's no $ involved.
//
//
// match_seq
// ---------
//
//     A matcher that matches a sequence of sub-matchers, denoted various
//     possible ways:
//
//             $(M)*       zero or more Ms
//             $(M)+       one or more Ms
//             $(M),+      one or more comma-separated Ms
//             $(A B C);*  zero or more semi-separated 'A B C' seqs
//
//
// match_nonterminal
// -----------------
//
//     A matcher that matches one of a few interesting named rust
//     nonterminals, such as types, expressions, items, or raw token-trees. A
//     black-box matcher on expr, for example, binds an expr to a given ident,
//     and that ident can re-occur as an interpolation in the RHS of a
//     macro-by-example rule. For example:
//
//        $foo:expr   =>     1 + $foo    // interpolate an expr
//        $foo:tt     =>     $foo        // interpolate a token-tree
//        $foo:tt     =>     bar! $foo   // only other valid interpolation
//                                       // is in arg position for another
//                                       // macro
//
// As a final, horrifying aside, note that macro-by-example's input is
// also matched by one of these matchers. Holy self-referential! It is matched
// by an match_seq, specifically this one:
//
//                   $( $lhs:matchers => $rhs:tt );+
//
// If you understand that, you have closed to loop and understand the whole
// macro system. Congratulations.
//
type matcher = spanned<matcher_>;

#[auto_encode]
#[auto_decode]
enum matcher_ {
    // match one token
    match_tok(::parse::token::Token),
    // match repetitions of a sequence: body, separator, zero ok?,
    // lo, hi position-in-match-array used:
    match_seq(~[matcher], Option<::parse::token::Token>, bool, uint, uint),
    // parse a Rust NT: name to bind, name of NT, position in match array:
    match_nonterminal(ident, ident, uint)
}

type mac = spanned<mac_>;

#[auto_encode]
#[auto_decode]
enum mac_ {
    mac_invoc_tt(@path,~[token_tree]),   // new macro-invocation
}

type lit = spanned<lit_>;

#[auto_encode]
#[auto_decode]
enum lit_ {
    lit_str(@~str),
    lit_int(i64, int_ty),
    lit_uint(u64, uint_ty),
    lit_int_unsuffixed(i64),
    lit_float(@~str, float_ty),
    lit_float_unsuffixed(@~str),
    lit_nil,
    lit_bool(bool),
}

impl lit_: cmp::Eq {
    pure fn eq(&self, other: &lit_) -> bool {
        match ((*self), *other) {
            (lit_str(a), lit_str(b)) => a == b,
            (lit_int(val_a, ty_a), lit_int(val_b, ty_b)) => {
                val_a == val_b && ty_a == ty_b
            }
            (lit_uint(val_a, ty_a), lit_uint(val_b, ty_b)) => {
                val_a == val_b && ty_a == ty_b
            }
            (lit_int_unsuffixed(a), lit_int_unsuffixed(b)) => a == b,
            (lit_float(val_a, ty_a), lit_float(val_b, ty_b)) => {
                val_a == val_b && ty_a == ty_b
            }
            (lit_float_unsuffixed(a), lit_float_unsuffixed(b)) => a == b,
            (lit_nil, lit_nil) => true,
            (lit_bool(a), lit_bool(b)) => a == b,
            (lit_str(_), _) => false,
            (lit_int(*), _) => false,
            (lit_uint(*), _) => false,
            (lit_int_unsuffixed(*), _) => false,
            (lit_float(*), _) => false,
            (lit_float_unsuffixed(*), _) => false,
            (lit_nil, _) => false,
            (lit_bool(_), _) => false
        }
    }
    pure fn ne(&self, other: &lit_) -> bool { !(*self).eq(other) }
}

// NB: If you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[auto_encode]
#[auto_decode]
struct mt {
    ty: @Ty,
    mutbl: mutability,
}

#[auto_encode]
#[auto_decode]
type ty_field_ = {ident: ident, mt: mt};

type ty_field = spanned<ty_field_>;

#[auto_encode]
#[auto_decode]
type ty_method = {ident: ident, attrs: ~[attribute], purity: purity,
                  decl: fn_decl, tps: ~[ty_param], self_ty: self_ty,
                  id: node_id, span: span};

#[auto_encode]
#[auto_decode]
// A trait method is either required (meaning it doesn't have an
// implementation, just a signature) or provided (meaning it has a default
// implementation).
enum trait_method {
    required(ty_method),
    provided(@method),
}

#[auto_encode]
#[auto_decode]
enum int_ty { ty_i, ty_char, ty_i8, ty_i16, ty_i32, ty_i64, }

impl int_ty : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl int_ty : cmp::Eq {
    pure fn eq(&self, other: &int_ty) -> bool {
        match ((*self), (*other)) {
            (ty_i, ty_i) => true,
            (ty_char, ty_char) => true,
            (ty_i8, ty_i8) => true,
            (ty_i16, ty_i16) => true,
            (ty_i32, ty_i32) => true,
            (ty_i64, ty_i64) => true,
            (ty_i, _) => false,
            (ty_char, _) => false,
            (ty_i8, _) => false,
            (ty_i16, _) => false,
            (ty_i32, _) => false,
            (ty_i64, _) => false,
        }
    }
    pure fn ne(&self, other: &int_ty) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum uint_ty { ty_u, ty_u8, ty_u16, ty_u32, ty_u64, }

impl uint_ty : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl uint_ty : cmp::Eq {
    pure fn eq(&self, other: &uint_ty) -> bool {
        match ((*self), (*other)) {
            (ty_u, ty_u) => true,
            (ty_u8, ty_u8) => true,
            (ty_u16, ty_u16) => true,
            (ty_u32, ty_u32) => true,
            (ty_u64, ty_u64) => true,
            (ty_u, _) => false,
            (ty_u8, _) => false,
            (ty_u16, _) => false,
            (ty_u32, _) => false,
            (ty_u64, _) => false
        }
    }
    pure fn ne(&self, other: &uint_ty) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum float_ty { ty_f, ty_f32, ty_f64, }

impl float_ty : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl float_ty : cmp::Eq {
    pure fn eq(&self, other: &float_ty) -> bool {
        match ((*self), (*other)) {
            (ty_f, ty_f) | (ty_f32, ty_f32) | (ty_f64, ty_f64) => true,
            (ty_f, _) | (ty_f32, _) | (ty_f64, _) => false
        }
    }
    pure fn ne(&self, other: &float_ty) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
struct Ty {
    id: node_id,
    node: ty_,
    span: span,
}

// Not represented directly in the AST, referred to by name through a ty_path.
#[auto_encode]
#[auto_decode]
enum prim_ty {
    ty_int(int_ty),
    ty_uint(uint_ty),
    ty_float(float_ty),
    ty_str,
    ty_bool,
}

impl prim_ty : cmp::Eq {
    pure fn eq(&self, other: &prim_ty) -> bool {
        match (*self) {
            ty_int(e0a) => {
                match (*other) {
                    ty_int(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_uint(e0a) => {
                match (*other) {
                    ty_uint(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_float(e0a) => {
                match (*other) {
                    ty_float(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ty_str => {
                match (*other) {
                    ty_str => true,
                    _ => false
                }
            }
            ty_bool => {
                match (*other) {
                    ty_bool => true,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &prim_ty) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
type region = {id: node_id, node: region_};

#[auto_encode]
#[auto_decode]
enum region_ {
    re_anon,
    re_static,
    re_self,
    re_named(ident)
}

#[auto_encode]
#[auto_decode]
#[deriving_eq]
enum Onceness {
    Once,
    Many
}

impl Onceness : ToStr {
    pure fn to_str() -> ~str {
        match self {
            Once => ~"once",
            Many => ~"many"
        }
    }
}

impl Onceness : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as uint).iter_bytes(lsb0, f);
    }
}

#[auto_encode]
#[auto_decode]
struct TyFn {
    proto: Proto,
    region: Option<@region>,
    purity: purity,
    onceness: Onceness,
    bounds: @~[ty_param_bound],
    decl: fn_decl
}

#[auto_encode]
#[auto_decode]
enum ty_ {
    ty_nil,
    ty_bot, /* bottom type */
    ty_box(mt),
    ty_uniq(mt),
    ty_vec(mt),
    ty_fixed_length_vec(mt, uint),
    ty_ptr(mt),
    ty_rptr(@region, mt),
    ty_rec(~[ty_field]),
    ty_fn(@TyFn),
    ty_tup(~[@Ty]),
    ty_path(@path, node_id),
    ty_mac(mac),
    // ty_infer means the type should be inferred instead of it having been
    // specified. This should only appear at the "top level" of a type and not
    // nested in one.
    ty_infer,
}

// Equality and byte-iter (hashing) can be quite approximate for AST types.
// since we only care about this for normalizing them to "real" types.
impl Ty : cmp::Eq {
    pure fn eq(&self, other: &Ty) -> bool {
        ptr::addr_of(&(*self)) == ptr::addr_of(&(*other))
    }
    pure fn ne(&self, other: &Ty) -> bool {
        ptr::addr_of(&(*self)) != ptr::addr_of(&(*other))
    }
}

impl Ty : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        to_bytes::iter_bytes_2(&self.span.lo, &self.span.hi, lsb0, f);
    }
}


#[auto_encode]
#[auto_decode]
type arg = {mode: mode, ty: @Ty, pat: @pat, id: node_id};

#[auto_encode]
#[auto_decode]
type fn_decl =
    {inputs: ~[arg],
     output: @Ty,
     cf: ret_style};

#[auto_encode]
#[auto_decode]
pub enum purity {
    pure_fn, // declared with "pure fn"
    unsafe_fn, // declared with "unsafe fn"
    impure_fn, // declared with "fn"
    extern_fn, // declared with "extern fn"
}

impl purity : ToStr {
    pure fn to_str() -> ~str {
        match self {
            impure_fn => ~"impure",
            unsafe_fn => ~"unsafe",
            pure_fn => ~"pure",
            extern_fn => ~"extern"
        }
    }
}

impl purity : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl purity : cmp::Eq {
    pure fn eq(&self, other: &purity) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &purity) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum ret_style {
    noreturn, // functions with return type _|_ that always
              // raise an error or exit (i.e. never return to the caller)
    return_val, // everything else
}

impl ret_style : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl ret_style : cmp::Eq {
    pure fn eq(&self, other: &ret_style) -> bool {
        match ((*self), (*other)) {
            (noreturn, noreturn) => true,
            (return_val, return_val) => true,
            (noreturn, _) => false,
            (return_val, _) => false,
        }
    }
    pure fn ne(&self, other: &ret_style) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
enum self_ty_ {
    sty_static,                         // no self: static method
    sty_by_ref,                         // old by-reference self: ``
    sty_value,                          // by-value self: `self`
    sty_region(mutability),             // by-region self: `&self`
    sty_box(mutability),                // by-managed-pointer self: `@self`
    sty_uniq(mutability)                // by-unique-pointer self: `~self`
}

impl self_ty_ : cmp::Eq {
    pure fn eq(&self, other: &self_ty_) -> bool {
        match (*self) {
            sty_static => {
                match (*other) {
                    sty_static => true,
                    _ => false
                }
            }
            sty_by_ref => {
                match (*other) {
                    sty_by_ref => true,
                    _ => false
                }
            }
            sty_value => {
                match (*other) {
                    sty_value => true,
                    _ => false
                }
            }
            sty_region(e0a) => {
                match (*other) {
                    sty_region(e0b) => e0a == e0b,
                    _ => false
                }
            }
            sty_box(e0a) => {
                match (*other) {
                    sty_box(e0b) => e0a == e0b,
                    _ => false
                }
            }
            sty_uniq(e0a) => {
                match (*other) {
                    sty_uniq(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(&self, other: &self_ty_) -> bool { !(*self).eq(other) }
}

type self_ty = spanned<self_ty_>;

#[auto_encode]
#[auto_decode]
type method = {ident: ident, attrs: ~[attribute],
               tps: ~[ty_param], self_ty: self_ty,
               purity: purity, decl: fn_decl, body: blk,
               id: node_id, span: span, self_id: node_id,
               vis: visibility};

#[auto_encode]
#[auto_decode]
type _mod = {view_items: ~[@view_item], items: ~[@item]};

#[auto_encode]
#[auto_decode]
enum foreign_abi {
    foreign_abi_rust_intrinsic,
    foreign_abi_cdecl,
    foreign_abi_stdcall,
}

// Foreign mods can be named or anonymous
#[auto_encode]
#[auto_decode]
enum foreign_mod_sort { named, anonymous }

impl foreign_mod_sort : cmp::Eq {
    pure fn eq(&self, other: &foreign_mod_sort) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &foreign_mod_sort) -> bool { !(*self).eq(other) }
}

impl foreign_abi : cmp::Eq {
    pure fn eq(&self, other: &foreign_abi) -> bool {
        match ((*self), (*other)) {
            (foreign_abi_rust_intrinsic, foreign_abi_rust_intrinsic) => true,
            (foreign_abi_cdecl, foreign_abi_cdecl) => true,
            (foreign_abi_stdcall, foreign_abi_stdcall) => true,
            (foreign_abi_rust_intrinsic, _) => false,
            (foreign_abi_cdecl, _) => false,
            (foreign_abi_stdcall, _) => false,
        }
    }
    pure fn ne(&self, other: &foreign_abi) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
type foreign_mod =
    {sort: foreign_mod_sort,
     abi: ident,
     view_items: ~[@view_item],
     items: ~[@foreign_item]};

#[auto_encode]
#[auto_decode]
type variant_arg = {ty: @Ty, id: node_id};

#[auto_encode]
#[auto_decode]
enum variant_kind {
    tuple_variant_kind(~[variant_arg]),
    struct_variant_kind(@struct_def),
    enum_variant_kind(enum_def)
}

#[auto_encode]
#[auto_decode]
type enum_def_ = { variants: ~[variant], common: Option<@struct_def> };

#[auto_encode]
#[auto_decode]
enum enum_def = enum_def_;

#[auto_encode]
#[auto_decode]
type variant_ = {name: ident, attrs: ~[attribute], kind: variant_kind,
                 id: node_id, disr_expr: Option<@expr>, vis: visibility};

type variant = spanned<variant_>;

#[auto_encode]
#[auto_decode]
struct path_list_ident_ {
    name: ident,
    id: node_id,
}

type path_list_ident = spanned<path_list_ident_>;

#[auto_encode]
#[auto_decode]
enum namespace { module_ns, type_value_ns }

impl namespace : cmp::Eq {
    pure fn eq(&self, other: &namespace) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &namespace) -> bool { !(*self).eq(other) }
}

type view_path = spanned<view_path_>;

#[auto_encode]
#[auto_decode]
enum view_path_ {

    // quux = foo::bar::baz
    //
    // or just
    //
    // foo::bar::baz  (with 'baz =' implicitly on the left)
    view_path_simple(ident, @path, namespace, node_id),

    // foo::bar::*
    view_path_glob(@path, node_id),

    // foo::bar::{a,b,c}
    view_path_list(@path, ~[path_list_ident], node_id)
}

#[auto_encode]
#[auto_decode]
struct view_item {
    node: view_item_,
    attrs: ~[attribute],
    vis: visibility,
    span: span,
}

#[auto_encode]
#[auto_decode]
enum view_item_ {
    view_item_use(ident, ~[@meta_item], node_id),
    view_item_import(~[@view_path]),
    view_item_export(~[@view_path])
}

// Meta-data associated with an item
type attribute = spanned<attribute_>;

// Distinguishes between attributes that decorate items and attributes that
// are contained as statements within items. These two cases need to be
// distinguished for pretty-printing.
#[auto_encode]
#[auto_decode]
enum attr_style { attr_outer, attr_inner, }

impl attr_style : cmp::Eq {
    pure fn eq(&self, other: &attr_style) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &attr_style) -> bool { !(*self).eq(other) }
}

// doc-comments are promoted to attributes that have is_sugared_doc = true
#[auto_encode]
#[auto_decode]
struct attribute_ {
    style: attr_style,
    value: meta_item,
    is_sugared_doc: bool,
}

/*
  trait_refs appear in impls.
  resolve maps each trait_ref's ref_id to its defining trait; that's all
  that the ref_id is for. The impl_id maps to the "self type" of this impl.
  If this impl is an item_impl, the impl_id is redundant (it could be the
  same as the impl's node id).
 */
#[auto_encode]
#[auto_decode]
type trait_ref = {path: @path, ref_id: node_id};

#[auto_encode]
#[auto_decode]
enum visibility { public, private, inherited }

impl visibility : cmp::Eq {
    pure fn eq(&self, other: &visibility) -> bool {
        match ((*self), (*other)) {
            (public, public) => true,
            (private, private) => true,
            (inherited, inherited) => true,
            (public, _) => false,
            (private, _) => false,
            (inherited, _) => false,
        }
    }
    pure fn ne(&self, other: &visibility) -> bool { !(*self).eq(other) }
}

#[auto_encode]
#[auto_decode]
struct struct_field_ {
    kind: struct_field_kind,
    id: node_id,
    ty: @Ty,
}

type struct_field = spanned<struct_field_>;

#[auto_encode]
#[auto_decode]
enum struct_field_kind {
    named_field(ident, struct_mutability, visibility),
    unnamed_field   // element of a tuple-like struct
}

impl struct_field_kind : cmp::Eq {
    pure fn eq(&self, other: &struct_field_kind) -> bool {
        match (*self) {
            named_field(ident_a, struct_mutability_a, visibility_a) => {
                match *other {
                    named_field(ident_b, struct_mutability_b, visibility_b)
                            => {
                        ident_a == ident_b &&
                        struct_mutability_a == struct_mutability_b &&
                        visibility_a == visibility_b
                    }
                    unnamed_field => false
                }
            }
            unnamed_field => {
                match *other {
                    named_field(*) => false,
                    unnamed_field => true
                }
            }
        }
    }
    pure fn ne(&self, other: &struct_field_kind) -> bool {
        !(*self).eq(other)
    }
}

#[auto_encode]
#[auto_decode]
struct struct_def {
    fields: ~[@struct_field], /* fields */
    /* (not including ctor or dtor) */
    /* dtor is optional */
    dtor: Option<struct_dtor>,
    /* ID of the constructor. This is only used for tuple- or enum-like
     * structs. */
    ctor_id: Option<node_id>
}

/*
  FIXME (#3300): Should allow items to be anonymous. Right now
  we just use dummy names for anon items.
 */
#[auto_encode]
#[auto_decode]
struct item {
    ident: ident,
    attrs: ~[attribute],
    id: node_id,
    node: item_,
    vis: visibility,
    span: span,
}

#[auto_encode]
#[auto_decode]
enum item_ {
    item_const(@Ty, @expr),
    item_fn(fn_decl, purity, ~[ty_param], blk),
    item_mod(_mod),
    item_foreign_mod(foreign_mod),
    item_ty(@Ty, ~[ty_param]),
    item_enum(enum_def, ~[ty_param]),
    item_struct(@struct_def, ~[ty_param]),
    item_trait(~[ty_param], ~[@trait_ref], ~[trait_method]),
    item_impl(~[ty_param],
              Option<@trait_ref>, /* (optional) trait this impl implements */
              @Ty, /* self */
              ~[@method]),
    item_mac(mac),
}

#[auto_encode]
#[auto_decode]
enum struct_mutability { struct_mutable, struct_immutable }

impl struct_mutability : to_bytes::IterBytes {
    pure fn iter_bytes(&self, +lsb0: bool, f: to_bytes::Cb) {
        (*self as u8).iter_bytes(lsb0, f)
    }
}

impl struct_mutability : cmp::Eq {
    pure fn eq(&self, other: &struct_mutability) -> bool {
        match ((*self), (*other)) {
            (struct_mutable, struct_mutable) => true,
            (struct_immutable, struct_immutable) => true,
            (struct_mutable, _) => false,
            (struct_immutable, _) => false,
        }
    }
    pure fn ne(&self, other: &struct_mutability) -> bool {
        !(*self).eq(other)
    }
}

type struct_dtor = spanned<struct_dtor_>;

#[auto_encode]
#[auto_decode]
struct struct_dtor_ {
    id: node_id,
    attrs: ~[attribute],
    self_id: node_id,
    body: blk,
}

#[auto_encode]
#[auto_decode]
struct foreign_item {
    ident: ident,
    attrs: ~[attribute],
    node: foreign_item_,
    id: node_id,
    span: span,
    vis: visibility,
}

#[auto_encode]
#[auto_decode]
enum foreign_item_ {
    foreign_item_fn(fn_decl, purity, ~[ty_param]),
    foreign_item_const(@Ty)
}

// The data we save and restore about an inlined item or method.  This is not
// part of the AST that we parse from a file, but it becomes part of the tree
// that we trans.
#[auto_encode]
#[auto_decode]
enum inlined_item {
    ii_item(@item),
    ii_method(def_id /* impl id */, @method),
    ii_foreign(@foreign_item),
    ii_dtor(struct_dtor, ident, ~[ty_param], def_id /* parent id */)
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
