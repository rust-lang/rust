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
import syntax::print::pprust;
import util::common::new_def_hash;

import std::list;
import std::list::list;
import std::map;
import std::map::hashmap;

type parent = option<ast::node_id>;

/* Records the parameter ID of a region name. */
type binding = {node_id: ast::node_id,
                name: str,
                br: ty::bound_region};

type region_map = {
    // Mapping from a block/function expression to its parent.
    parents: hashmap<ast::node_id,ast::node_id>,

    // Mapping from arguments and local variables to the block in
    // which they are declared. Arguments are considered to be declared
    // within the body of the function.
    local_blocks: hashmap<ast::node_id,ast::node_id>
};

type ctxt = {
    sess: session,
    def_map: resolve::def_map,
    region_map: @region_map,

    // These two fields (parent and closure_parent) specify the parent
    // scope of the current expression.  The parent scope is the
    // innermost block, call, or alt expression during the execution
    // of which the current expression will be evaluated.  Generally
    // speaking, the innermost parent scope is also the closest
    // suitable ancestor in the AST tree.
    //
    // However, there are two subtle cases where the parent scope for
    // an expression is not strictly derived from the AST. The first
    // such exception concerns call arguments and the second concerns
    // closures (which, at least today, are always call arguments).
    // Consider:
    //
    // { // block a
    //    foo( // call b
    //        x,
    //        y,
    //        fn&() {
    //          // fn body c
    //        })
    // }
    //
    // Here, the parent of the three argument expressions is
    // actually the block `a`, not the call `b`, because they will
    // be evaluated before the call conceptually takes place.
    // However, the body of the closure is parented by the call
    // `b` (it cannot be invoked except during that call, after
    // all).
    //
    // To capture these patterns, we use two fields.  The first,
    // parent, is the parent scope of a normal expression.  The
    // second, closure_parent, is the parent scope that a closure body
    // ought to use.  These only differ in the case of calls, where
    // the closure parent is the call, but the parent is the container
    // of the call.
    parent: parent,
    closure_parent: parent
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

    if a_ancestors[a_index] != b_ancestors[b_index] {
        ret none;
    }

    loop {
        // Loop invariant: a_ancestors[a_index] == b_ancestors[b_index]
        // for all indices between a_index and the end of the array
        if a_index == 0u { ret some(scope_a); }
        if b_index == 0u { ret some(scope_b); }
        a_index -= 1u;
        b_index -= 1u;
        if a_ancestors[a_index] != b_ancestors[b_index] {
            ret some(a_ancestors[a_index + 1u]);
        }
    }
}

fn parent_id(cx: ctxt, span: span) -> ast::node_id {
    alt cx.parent {
      none {
        cx.sess.span_bug(span, "crate should not be parent here");
      }
      some(parent_id) {
        parent_id
      }
    }
}

fn record_parent(cx: ctxt, child_id: ast::node_id) {
    alt cx.parent {
      none { /* no-op */ }
      some(parent_id) {
        #debug["parent of node %d is node %d", child_id, parent_id];
        cx.region_map.parents.insert(child_id, parent_id);
      }
    }
}

fn resolve_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Record the parent of this block.
    record_parent(cx, blk.node.id);

    // Descend.
    let new_cx: ctxt = {parent: some(blk.node.id),
                        closure_parent: some(blk.node.id) with cx};
    visit::visit_block(blk, new_cx, visitor);
}

fn resolve_arm(arm: ast::arm, cx: ctxt, visitor: visit::vt<ctxt>) {
    visit::visit_arm(arm, cx, visitor);
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
            /* This names a local. Bind it to the containing scope. */
            let local_blocks = cx.region_map.local_blocks;
            local_blocks.insert(pat.id, parent_id(cx, pat.span));
          }
        }
      }
      _ { /* no-op */ }
    }

    visit::visit_pat(pat, cx, visitor);
}

fn resolve_expr(expr: @ast::expr, cx: ctxt, visitor: visit::vt<ctxt>) {
    record_parent(cx, expr.id);
    alt expr.node {
      ast::expr_call(*) {
        #debug["node %d: %s", expr.id, pprust::expr_to_str(expr)];
        let new_cx = {closure_parent: some(expr.id) with cx};
        visit::visit_expr(expr, new_cx, visitor);
      }
      ast::expr_alt(subexpr, _, _) {
        #debug["node %d: %s", expr.id, pprust::expr_to_str(expr)];
        let new_cx = {parent: some(expr.id),
                      closure_parent: some(expr.id)
                      with cx};
        visit::visit_expr(expr, new_cx, visitor);
      }
      ast::expr_fn(_, _, _, cap_clause) |
      ast::expr_fn_block(_, _, cap_clause) {
        // although the capture items are not expressions per se, they
        // do get "evaluated" in some sense as copies or moves of the
        // relevant variables so we parent them like an expression
        for (*cap_clause).each { |cap_item|
            record_parent(cx, cap_item.id);
        }
        visit::visit_expr(expr, cx, visitor);
      }
      _ {
        visit::visit_expr(expr, cx, visitor);
      }
    }
}

fn resolve_local(local: @ast::local, cx: ctxt, visitor: visit::vt<ctxt>) {
    cx.region_map.local_blocks.insert(
        local.node.id, parent_id(cx, local.span));
    visit::visit_local(local, cx, visitor);
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx: ctxt = {closure_parent: some(item.id),
                        parent: some(item.id) with cx};
    visit::visit_item(item, new_cx, visitor);
}

fn resolve_fn(fk: visit::fn_kind, decl: ast::fn_decl, body: ast::blk,
              sp: span, id: ast::node_id, cx: ctxt,
              visitor: visit::vt<ctxt>) {

    let fn_cx = alt fk {
      visit::fk_item_fn(*) | visit::fk_method(*) | visit::fk_res(*) |
      visit::fk_ctor(*) | visit::fk_dtor(*) {
        // Top-level functions are a root scope.
        {parent: some(id), closure_parent: some(id) with cx}
      }

      visit::fk_anon(*) | visit::fk_fn_block(*) {
        // Closures use the closure_parent.
        {parent: cx.closure_parent with cx}
      }
    };

    #debug["visiting fn with body %d. cx.parent: %? \
            cx.closure_parent: %? fn_cx.parent: %?",
           body.node.id, cx.parent,
           cx.closure_parent, fn_cx.parent];

    for decl.inputs.each { |input|
        cx.region_map.local_blocks.insert(
            input.id, body.node.id);
    }

    visit::visit_fn(fk, decl, body, sp, id, fn_cx, visitor);
}

fn resolve_crate(sess: session, def_map: resolve::def_map, crate: @ast::crate)
        -> @region_map {
    let cx: ctxt = {sess: sess,
                    def_map: def_map,
                    region_map: @{parents: map::int_hash(),
                                  local_blocks: map::int_hash()},
                    parent: none,
                    closure_parent: none};
    let visitor = visit::mk_vt(@{
        visit_block: resolve_block,
        visit_item: resolve_item,
        visit_fn: resolve_fn,
        visit_arm: resolve_arm,
        visit_pat: resolve_pat,
        visit_expr: resolve_expr,
        visit_local: resolve_local
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);
    ret cx.region_map;
}

