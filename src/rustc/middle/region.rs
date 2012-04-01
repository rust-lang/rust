/*

Region resolution. This pass runs before typechecking and resolves region
names to the appropriate block.

This seems to be as good a place as any to explain in detail how
region naming, representation, and type check works.

### Naming and so forth

We really want regions to be very lightweight to use. Therefore,
unlike other named things, the scopes for regions are not explicitly
declared: instead, they are implicitly defined.  Functions declare new
scopes: if the function is not a bare function, then as always it
inherits the names in scope from the outer scope.  Within a function
declaration, new names implicitly declare new region variables.  Outside
of function declarations, new names are illegal.  To make this more
concrete, here is an example:

    fn foo(s: &a.S, t: &b.T) {
        let s1: &a.S = s; // a refers to the same a as in the decl
        let t1: &c.T = t; // illegal: cannot introduce new name here
    }

The code in this file is what actually handles resolving these names.
It creates a couple of maps that map from the AST node representing a
region ptr type to the resolved form of its region parameter.  If new
names are introduced where they shouldn't be, then an error is
reported.

If regions are not given an explicit name, then the behavior depends
a bit on the context.  Within a function declaration, all unnamed regions
are mapped to a single, anonymous parameter.  That is, a function like:

    fn foo(s: &S) -> &S { s }

is equivalent to a declaration like:

    fn foo(s: &a.S) -> &a.S { s }

Within a function body or other non-binding context, an unnamed region
reference is mapped to a fresh region variable whose value can be
inferred as normal.

The resolved form of regions is `ty::region`.  Before I can explain
why this type is setup the way it is, I have to digress a little bit
into some ill-explained type theory.

### Universal Quantification

Regions are more complex than type parameters because, unlike type
parameters, they can be universally quantified within a type.  To put
it another way, you cannot (at least at the time of this writing) have
a variable `x` of type `fn<T>(T) -> T`.  You can have an *item* of
type `fn<T>(T) - T`, but whenever it is referenced within a method,
that type parameter `T` is replaced with a concrete type *variable*
`$T`.  To make this more concrete, imagine this code:

    fn identity<T>(x: T) -> T { x }
    let f = identity; // f has type fn($T) -> $T
    f(3u); // $T is bound to uint
    f(3);  // Type error

You can see here that a type error will result because the type of `f`
(as opposed to the type of `identity`) is not universally quantified
over `$T`.  That's fancy math speak for saying that the type variable
`$T` refers to a specific type that may not yet be known, unlike the
type parameter `T` which refers to some type which will never be
known.

Anyway, regions work differently.  If you have an item of type
`fn(&a.T) -> &a.T` and you reference it, its type remains the same:
only when the function *is called* is `&a` instantiated with a
concrete region variable.  This means you could call it twice and give
different values for `&a` each time.

This more general form is possible for regions because they do not
impact code generation.  We do not need to monomorphize functions
differently just because they contain region pointers.  In fact, we
don't really do *anything* differently.

### Representing regions; or, why do I care about all that?

The point of this discussion is that the representation of regions
must distinguish between a *bound* reference to a region and a *free*
reference.  A bound reference is one which will be replaced with a
fresh type variable when the function is called, like the type
parameter `T` in `identity`.  They can only appear within function
types.  A free reference is a region that may not yet be concretely
known, like the variable `$T`.

To see why we must distinguish them carefully, consider this program:

    fn item1(s: &a.S) {
        let choose = fn@(s1: &a.S) -> &a.S {
            if some_cond { s } else { s1 }
        };
    }

Here, the variable `s1: &a.S` that appears within the `fn@` is a free
reference to `a`.  That is, when you call `choose()`, you don't
replace `&a` with a fresh region variable, but rather you expect `s1`
to be in the same region as the parameter `s`.

But in this program, this is not the case at all:

    fn item2() {
        let identity = fn@(s1: &a.S) -> &a.S { s1 };
    }

To distinguish between these two cases, `ty::region` contains two
variants: `re_bound` and `re_free`.  In `item1()`, the outer reference
to `&a` would be `re_bound(rid_param("a", 0u))`, and the inner reference
would be `re_free(rid_param("a", 0u))`.  In `item2()`, the inner reference
would be `re_bound(rid_param("a", 0u))`.

#### Impliciations for typeck

In typeck, whenever we call a function, we must go over and replace
all references to `re_bound()` regions within its parameters with
fresh type variables (we do not, however, replace bound regions within
nested function types, as those nested functions have not yet been
called).

Also, when we typecheck the *body* of an item, we must replace all
`re_bound` references with `re_free` references.  This means that the
region in the type of the argument `s` in `item1()` *within `item1()`*
is not `re_bound(re_param("a", 0u))` but rather `re_free(re_param("a",
0u))`.  This is because, for any particular *invocation of `item1()`*,
`&a` will be bound to some specific region, and hence it is no longer
bound.

*/

import driver::session::session;
import middle::ty;
import syntax::{ast, visit};
import syntax::codemap::span;
import util::common::new_def_hash;

import std::list;
import std::list::list;
import std::map;
import std::map::hashmap;

/* Represents the type of the most immediate parent node. */
enum parent {
    pa_fn_item(ast::node_id),
    pa_block(ast::node_id),
    pa_nested_fn(ast::node_id),
    pa_item(ast::node_id),
    pa_crate
}

/* Records the parameter ID of a region name. */
type binding = {node_id: ast::node_id,
                name: str,
                br: ty::bound_region};

type region_map = {
    /* Mapping from a block/function expression to its parent. */
    parents: hashmap<ast::node_id,ast::node_id>,
    /* Mapping from a region type in the AST to its resolved region. */
    ast_type_to_region: hashmap<ast::node_id,ty::region>,
    /* Mapping from a local variable to its containing block. */
    local_blocks: hashmap<ast::node_id,ast::node_id>,
    /* Mapping from an AST type node to the region that `&` resolves to. */
    ast_type_to_inferred_region: hashmap<ast::node_id,ty::region>,
    /*
     * Mapping from an address-of operator or alt expression to its containing
     * block. This is used as the region if the operand is an rvalue.
     */
    rvalue_to_block: hashmap<ast::node_id,ast::node_id>
};

type region_scope = @{
    node_id: ast::node_id,
    kind: region_scope_kind
};

enum region_scope_kind {
    rsk_root,
    rsk_body(region_scope),
    rsk_self(region_scope),
    rsk_binding(region_scope, @mut [binding])
}

fn root_scope(node_id: ast::node_id) -> region_scope {
    @{node_id: node_id, kind: rsk_root}
}

impl methods for region_scope {
    fn body_subscope(node_id: ast::node_id) -> region_scope {
        @{node_id: node_id, kind: rsk_body(self)}
    }

    fn binding_subscope(node_id: ast::node_id) -> region_scope {
        @{node_id: node_id, kind: rsk_binding(self, @mut [])}
    }

    fn self_subscope(node_id: ast::node_id) -> region_scope {
        @{node_id: node_id, kind: rsk_self(self)}
    }

    fn find(nm: str) -> option<binding> {
        alt self.kind {
          rsk_root { none }
          rsk_body(parent) { parent.find(nm) }
          rsk_self(parent) { parent.find(nm) }
          rsk_binding(parent, bs) {
            alt (*bs).find({|b| b.name == nm }) {
              none { parent.find(nm) }
              some(b) { some(b) }
            }
          }
        }
    }

    // fn resolve_anon() -> option<ty::region> {
    //     alt self.kind {
    //       rsk_root { none }
    //       rsk_body(_) { none }
    //       rsk_self(_) { none }
    //       rsk_binding(_, _) { ty::re_bound(ty::br_anon) }
    //     }
    // }

    fn resolve_self_helper(bound: bool) -> option<ty::region> {
        alt self.kind {
          rsk_root { none }
          rsk_self(_) if bound { some(ty::re_bound(ty::br_self)) }
          rsk_self(_) { some(ty::re_free(self.node_id, ty::br_self)) }
          rsk_binding(p, _) { p.resolve_self_helper(bound) }
          rsk_body(p) { p.resolve_self_helper(false) }
        }
    }

    fn resolve_self() -> option<ty::region> {
        self.resolve_self_helper(true)
    }

    fn resolve_ident(nm: str) -> option<ty::region> {
        alt self.find(nm) {
          some(b) if b.node_id == self.node_id {
            some(ty::re_bound(b.br))
          }

          some(b) {
            some(ty::re_free(b.node_id, b.br))
          }

          none {
            alt self.kind {
              rsk_self(_) | rsk_root | rsk_body(_) { none }
              rsk_binding(_, bs) {
                let idx = (*bs).len();
                let br = ty::br_param(idx, nm);
                vec::push(*bs, {node_id: self.node_id,
                                name: nm,
                                br: br});
                some(ty::re_bound(br))
              }
            }
          }
        }
    }
}

type ctxt = {
    sess: session,
    def_map: resolve::def_map,
    region_map: @region_map,

    scope: region_scope,

    /*
     * A list of local IDs that will be parented to the next block we
     * traverse. This is used when resolving `alt` statements. Since we see
     * the pattern before the associated block, upon seeing a pattern we must
     * parent all the bindings in that pattern to the next block we see.
     */
    mut queued_locals: [ast::node_id],

    parent: parent,

    /* True if we're within the pattern part of an alt, false otherwise. */
    in_alt: bool,

    /* The next parameter ID. */
    mut next_param_id: uint
};

// Returns true if `subscope` is equal to or is lexically nested inside
// `superscope` and false otherwise.
fn scope_contains(region_map: @region_map, superscope: ast::node_id,
                  subscope: ast::node_id) -> bool {
    let mut subscope = subscope;
    while superscope != subscope {
        alt region_map.parents.find(subscope) {
            none { ret false; }
            some(scope) { subscope = scope; }
        }
    }
    ret true;
}

fn nearest_common_ancestor(region_map: @region_map, scope_a: ast::node_id,
                           scope_b: ast::node_id) -> option<ast::node_id> {

    fn ancestors_of(region_map: @region_map, scope: ast::node_id)
                    -> [ast::node_id] {
        let mut result = [scope];
        let mut scope = scope;
        loop {
            alt region_map.parents.find(scope) {
                none { ret result; }
                some(superscope) {
                    result += [superscope];
                    scope = superscope;
                }
            }
        }
    }

    if scope_a == scope_b { ret some(scope_a); }

    let a_ancestors = ancestors_of(region_map, scope_a);
    let b_ancestors = ancestors_of(region_map, scope_b);
    let mut a_index = vec::len(a_ancestors) - 1u;
    let mut b_index = vec::len(b_ancestors) - 1u;

    // Here, [ab]_ancestors is a vector going from narrow to broad.
    // The end of each vector will be the item where the scope is
    // defined; if there are any common ancestors, then the tails of
    // the vector will be the same.  So basically we want to walk
    // backwards from the tail of each vector and find the first point
    // where they diverge.  If one vector is a suffix of the other,
    // then the corresponding scope is a superscope of the other.

    loop {
        if a_ancestors[a_index] != b_ancestors[b_index] {
            if a_index == a_ancestors.len() {
                ret none;
            } else {
                ret some(a_ancestors[a_index + 1u]);
            }
        }
        if a_index == 0u { ret some(scope_a); }
        if b_index == 0u { ret some(scope_b); }
        a_index -= 1u;
        b_index -= 1u;
    }
}

fn get_inferred_region(cx: ctxt, sp: syntax::codemap::span) -> ty::region {
    // We infer to the caller region if we're at item scope
    // and to the block region if we're at block scope.
    //
    // TODO: What do we do if we're in an alt?

    ret alt cx.parent {
      pa_fn_item(_) | pa_nested_fn(_) { ty::re_bound(ty::br_anon) }
      pa_block(block_id) { ty::re_scope(block_id) }
      pa_item(_) { ty::re_bound(ty::br_anon) }
      pa_crate { cx.sess.span_bug(sp, "inferred region at crate level?!"); }
    }
}

fn resolve_region_binding(cx: ctxt, span: span, region: ast::region) {

    let id = region.id;
    let rm = cx.region_map;
    alt region.node {
      ast::re_inferred {
        // option::may(cx.scope.resolve_anon()) {|r|
        //     rm.ast_type_to_region.insert(id, r);
        // }
      }

      ast::re_named(ident) {
        alt cx.scope.resolve_ident(ident) {
          some(r) {
            rm.ast_type_to_region.insert(id, r);
          }

          none {
            cx.sess.span_err(
                span,
                #fmt["the region `%s` is not declared", ident]);
          }
        }
      }

      ast::re_self {
        alt cx.scope.resolve_self() {
          some(r) {
            rm.ast_type_to_region.insert(id, r);
          }

          none {
            cx.sess.span_err(
                span,
                "the `self` region is not allowed here");
          }
        }
      }
    }
}

fn resolve_ty(ty: @ast::ty, cx: ctxt, visitor: visit::vt<ctxt>) {
    let inferred_region = get_inferred_region(cx, ty.span);
    cx.region_map.ast_type_to_inferred_region.insert(ty.id, inferred_region);

    alt ty.node {
      ast::ty_rptr(r, _) {
        resolve_region_binding(cx, ty.span, r);
      }
      _ { /* nothing to do */ }
    }

    visit::visit_ty(ty, cx, visitor);
}

fn record_parent(cx: ctxt, child_id: ast::node_id) {
    alt cx.parent {
        pa_fn_item(parent_id) |
        pa_item(parent_id) |
        pa_block(parent_id) |
        pa_nested_fn(parent_id) {
            cx.region_map.parents.insert(child_id, parent_id);
        }
        pa_crate { /* no-op */ }
    }
}

fn resolve_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Record the parent of this block.
    record_parent(cx, blk.node.id);

    // Resolve queued locals to this block.
    for local_id in cx.queued_locals {
        cx.region_map.local_blocks.insert(local_id, blk.node.id);
    }

    // Descend.
    let new_cx: ctxt = {parent: pa_block(blk.node.id),
                        scope: cx.scope.body_subscope(blk.node.id),
                        mut queued_locals: [],
                        in_alt: false with cx};
    visit::visit_block(blk, new_cx, visitor);
}

fn resolve_arm(arm: ast::arm, cx: ctxt, visitor: visit::vt<ctxt>) {
    let new_cx: ctxt = {mut queued_locals: [], in_alt: true with cx};
    visit::visit_arm(arm, new_cx, visitor);
}

fn resolve_pat(pat: @ast::pat, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt pat.node {
        ast::pat_ident(path, _) {
            let defn_opt = cx.def_map.find(pat.id);
            alt defn_opt {
                some(ast::def_variant(_,_)) {
                    /* Nothing to do; this names a variant. */
                }
                _ {
                    /*
                     * This names a local. Enqueue it or bind it to the
                     * containing block, depending on whether we're in an alt
                     * or not.
                     */
                    if cx.in_alt {
                        vec::push(cx.queued_locals, pat.id);
                    } else {
                        alt cx.parent {
                            pa_block(block_id) {
                                let local_blocks = cx.region_map.local_blocks;
                                local_blocks.insert(pat.id, block_id);
                            }
                            _ {
                                cx.sess.span_bug(pat.span,
                                                 "unexpected parent");
                            }
                        }
                    }
                }
            }
        }
        _ { /* no-op */ }
    }

    visit::visit_pat(pat, cx, visitor);
}

fn resolve_expr(expr: @ast::expr, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt expr.node {
        ast::expr_fn(_, _, _, _) | ast::expr_fn_block(_, _) {
            record_parent(cx, expr.id);
            let new_cx = {parent: pa_nested_fn(expr.id),
                          scope: cx.scope.binding_subscope(expr.id),
                          in_alt: false with cx};
            visit::visit_expr(expr, new_cx, visitor);
        }
        ast::expr_addr_of(_, subexpr) | ast::expr_alt(subexpr, _, _) {
            // Record the block that this expression appears in, in case the
            // operand is an rvalue.
            alt cx.parent {
                pa_block(blk_id) {
                    cx.region_map.rvalue_to_block.insert(subexpr.id, blk_id);
                }
                _ { cx.sess.span_bug(expr.span, "expr outside of block?!"); }
            }
            visit::visit_expr(expr, cx, visitor);
        }
        _ { visit::visit_expr(expr, cx, visitor); }
    }
}

fn resolve_local(local: @ast::local, cx: ctxt, visitor: visit::vt<ctxt>) {
    alt cx.parent {
        pa_block(blk_id) {
            cx.region_map.rvalue_to_block.insert(local.node.id, blk_id);
        }
        _ { cx.sess.span_bug(local.span, "local outside of block?!"); }
    }
    visit::visit_local(local, cx, visitor);
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let {parent, scope} = {
        alt item.node {
          ast::item_fn(_, _, _) | ast::item_enum(_, _) {
            {parent: pa_fn_item(item.id),
             scope: cx.scope.binding_subscope(item.id)}
          }
          ast::item_impl(_, _, _, _) | ast::item_class(_, _, _) {
            {parent: pa_item(item.id),
             scope: cx.scope.self_subscope(item.id)}
          }
          _ {
            {parent: pa_item(item.id),
             scope: root_scope(item.id)}
          }
        }
    };

    let new_cx: ctxt = {parent: parent,
                        scope: scope,
                        in_alt: false,
                        mut next_param_id: 0u
                        with cx};

    visit::visit_item(item, new_cx, visitor);
}

fn resolve_crate(sess: session, def_map: resolve::def_map, crate: @ast::crate)
        -> @region_map {
    let cx: ctxt = {sess: sess,
                    def_map: def_map,
                    region_map: @{parents: map::int_hash(),
                                  ast_type_to_region: map::int_hash(),
                                  local_blocks: map::int_hash(),
                                  ast_type_to_inferred_region:
                                    map::int_hash(),
                                  rvalue_to_block: map::int_hash()},
                    scope: root_scope(0),
                    mut queued_locals: [],
                    parent: pa_crate,
                    in_alt: false,
                    mut next_param_id: 0u};
    let visitor = visit::mk_vt(@{
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_ty: resolve_ty,
        visit_arm: resolve_arm,
        visit_pat: resolve_pat,
        visit_expr: resolve_expr,
        visit_local: resolve_local
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
    ret cx.region_map;
}

