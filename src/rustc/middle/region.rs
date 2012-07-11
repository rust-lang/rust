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
type `fn<T>(T) -> T`, but whenever it is referenced within a method,
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

#### Implications for typeck

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
import syntax::ast_util::new_def_hash;
import syntax::ast_map;
import dvec::{dvec, extensions};
import metadata::csearch;

import std::list;
import std::list::list;
import std::map::{hashmap, int_hash};

type parent = option<ast::node_id>;

/* Records the parameter ID of a region name. */
type binding = {node_id: ast::node_id,
                name: str,
                br: ty::bound_region};

// Mapping from a block/expr/binding to the innermost scope that
// bounds its lifetime.  For a block/expression, this is the lifetime
// in which it will be evaluated.  For a binding, this is the lifetime
// in which is in scope.
type region_map = hashmap<ast::node_id, ast::node_id>;

type ctxt = {
    sess: session,
    def_map: resolve::def_map,
    region_map: region_map,

    // The parent scope is the innermost block, call, or alt
    // expression during the execution of which the current expression
    // will be evaluated.  Generally speaking, the innermost parent
    // scope is also the closest suitable ancestor in the AST tree.
    //
    // There is a subtle point concerning call arguments.  Imagine
    // you have a call:
    //
    // { // block a
    //     foo( // call b
    //        x,
    //        y);
    // }
    //
    // In what lifetime are the expressions `x` and `y` evaluated?  At
    // first, I imagine the answer was the block `a`, as the arguments
    // are evaluated before the call takes place.  But this turns out
    // to be wrong.  The lifetime of the call must encompass the
    // argument evaluation as well.
    //
    // The reason is that evaluation of an earlier argument could
    // create a borrow which exists during the evaluation of later
    // arguments.  Consider this torture test, for example,
    //
    // fn test1(x: @mut ~int) {
    //     foo(&**x, *x = ~5);
    // }
    //
    // Here, the first argument `&**x` will be a borrow of the `~int`,
    // but the second argument overwrites that very value! Bad.
    // (This test is borrowck-pure-scope-in-call.rs, btw)
    parent: parent
};

// Returns true if `subscope` is equal to or is lexically nested inside
// `superscope` and false otherwise.
fn scope_contains(region_map: region_map, superscope: ast::node_id,
                  subscope: ast::node_id) -> bool {
    let mut subscope = subscope;
    while superscope != subscope {
        alt region_map.find(subscope) {
            none { ret false; }
            some(scope) { subscope = scope; }
        }
    }
    ret true;
}

fn nearest_common_ancestor(region_map: region_map, scope_a: ast::node_id,
                           scope_b: ast::node_id) -> option<ast::node_id> {

    fn ancestors_of(region_map: region_map, scope: ast::node_id)
                    -> ~[ast::node_id] {
        let mut result = ~[scope];
        let mut scope = scope;
        loop {
            alt region_map.find(scope) {
                none { ret result; }
                some(superscope) {
                    vec::push(result, superscope);
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

    // Here, ~[ab]_ancestors is a vector going from narrow to broad.
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
        cx.region_map.insert(child_id, parent_id);
      }
    }
}

fn resolve_block(blk: ast::blk, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Record the parent of this block.
    record_parent(cx, blk.node.id);

    // Descend.
    let new_cx: ctxt = {parent: some(blk.node.id) with cx};
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
            record_parent(cx, pat.id);
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
        let new_cx = {parent: some(expr.id) with cx};
        visit::visit_expr(expr, new_cx, visitor);
      }
      ast::expr_alt(subexpr, _, _) {
        #debug["node %d: %s", expr.id, pprust::expr_to_str(expr)];
        let new_cx = {parent: some(expr.id) with cx};
        visit::visit_expr(expr, new_cx, visitor);
      }
      ast::expr_fn(_, _, _, cap_clause) |
      ast::expr_fn_block(_, _, cap_clause) {
        // although the capture items are not expressions per se, they
        // do get "evaluated" in some sense as copies or moves of the
        // relevant variables so we parent them like an expression
        for (*cap_clause).each |cap_item| {
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
    record_parent(cx, local.node.id);
    visit::visit_local(local, cx, visitor);
}

fn resolve_item(item: @ast::item, cx: ctxt, visitor: visit::vt<ctxt>) {
    // Items create a new outer block scope as far as we're concerned.
    let new_cx: ctxt = {parent: none with cx};
    visit::visit_item(item, new_cx, visitor);
}

fn resolve_fn(fk: visit::fn_kind, decl: ast::fn_decl, body: ast::blk,
              sp: span, id: ast::node_id, cx: ctxt,
              visitor: visit::vt<ctxt>) {

    let fn_cx = alt fk {
      visit::fk_item_fn(*) | visit::fk_method(*) |
      visit::fk_ctor(*) | visit::fk_dtor(*) {
        // Top-level functions are a root scope.
        {parent: some(id) with cx}
      }

      visit::fk_anon(*) | visit::fk_fn_block(*) {
        // Closures continue with the inherited scope.
        cx
      }
    };

    #debug["visiting fn with body %d. cx.parent: %? \
            fn_cx.parent: %?",
           body.node.id, cx.parent, fn_cx.parent];

    for decl.inputs.each |input| {
        cx.region_map.insert(input.id, body.node.id);
    }

    visit::visit_fn(fk, decl, body, sp, id, fn_cx, visitor);
}

fn resolve_crate(sess: session, def_map: resolve::def_map, crate: @ast::crate)
        -> region_map {
    let cx: ctxt = {sess: sess,
                    def_map: def_map,
                    region_map: int_hash(),
                    parent: none};
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

// ___________________________________________________________________________
// Determining region parameterization
//
// Infers which type defns must be region parameterized---this is done
// by scanning their contents to see whether they reference a region
// type, directly or indirectly.  This is a fixed-point computation.
//
// We do it in two passes.  First we walk the AST and construct a map
// from each type defn T1 to other defns which make use of it.  For example,
// if we have a type like:
//
//    type S = *int;
//    type T = S;
//
// Then there would be a map entry from S to T.  During the same walk,
// we also construct add any types that reference regions to a set and
// a worklist.  We can then process the worklist, propagating indirect
// dependencies until a fixed point is reached.

type region_paramd_items = hashmap<ast::node_id, ()>;
type dep_map = hashmap<ast::node_id, @dvec<ast::node_id>>;

type determine_rp_ctxt = @{
    sess: session,
    ast_map: ast_map::map,
    def_map: resolve::def_map,
    region_paramd_items: region_paramd_items,
    dep_map: dep_map,
    worklist: dvec<ast::node_id>,

    mut item_id: ast::node_id,
    mut anon_implies_rp: bool
};

impl methods for determine_rp_ctxt {
    fn add_rp(id: ast::node_id) {
        assert id != 0;
        if self.region_paramd_items.insert(id, ()) {
            #debug["add region-parameterized item: %d (%s)",
                   id, ast_map::node_id_to_str(self.ast_map, id)];
            self.worklist.push(id);
        } else {
            #debug["item %d already region-parameterized", id];
        }
    }

    fn add_dep(from: ast::node_id, to: ast::node_id) {
        #debug["add dependency from %d -> %d (%s -> %s)",
               from, to,
               ast_map::node_id_to_str(self.ast_map, from),
               ast_map::node_id_to_str(self.ast_map, to)];
        let vec = alt self.dep_map.find(from) {
            some(vec) => {vec}
            none => {
                let vec = @dvec();
                self.dep_map.insert(from, vec);
                vec
            }
        };
        if !vec.contains(to) { vec.push(to); }
    }

    fn region_is_relevant(r: @ast::region) -> bool {
        alt r.node {
          ast::re_anon {self.anon_implies_rp}
          ast::re_named(@"self") {true}
          ast::re_named(_) {false}
        }
    }

    fn with(item_id: ast::node_id, anon_implies_rp: bool, f: fn()) {
        let old_item_id = self.item_id;
        let old_anon_implies_rp = self.anon_implies_rp;
        self.item_id = item_id;
        self.anon_implies_rp = anon_implies_rp;
        #debug["with_item_id(%d, %b)", item_id, anon_implies_rp];
        let _i = util::common::indenter();
        f();
        self.item_id = old_item_id;
        self.anon_implies_rp = old_anon_implies_rp;
    }
}

fn determine_rp_in_item(item: @ast::item,
                        &&cx: determine_rp_ctxt,
                        visitor: visit::vt<determine_rp_ctxt>) {
    do cx.with(item.id, true) {
        visit::visit_item(item, cx, visitor);
    }
}

fn determine_rp_in_fn(fk: visit::fn_kind,
                      decl: ast::fn_decl,
                      body: ast::blk,
                      sp: span,
                      id: ast::node_id,
                      &&cx: determine_rp_ctxt,
                      visitor: visit::vt<determine_rp_ctxt>) {
    do cx.with(cx.item_id, false) {
        visit::visit_fn(fk, decl, body, sp, id, cx, visitor);
    }
}

fn determine_rp_in_ty_method(ty_m: ast::ty_method,
                             &&cx: determine_rp_ctxt,
                             visitor: visit::vt<determine_rp_ctxt>) {
    do cx.with(cx.item_id, false) {
        visit::visit_ty_method(ty_m, cx, visitor);
    }
}

fn determine_rp_in_ty(ty: @ast::ty,
                      &&cx: determine_rp_ctxt,
                      visitor: visit::vt<determine_rp_ctxt>) {

    // we are only interesting in types that will require an item to
    // be region-parameterized.  if cx.item_id is zero, then this type
    // is not a member of a type defn nor is it a constitutent of an
    // impl etc.  So we can ignore it and its components.
    if cx.item_id == 0 { ret; }

    // if this type directly references a region, either via a
    // region pointer like &r.ty or a region-parameterized path
    // like path/r, add to the worklist/set
    alt ty.node {
      ast::ty_rptr(r, _) |
      ast::ty_path(@{rp: some(r), _}, _) |
      ast::ty_vstore(_, ast::vstore_slice(r)) => {
        #debug["referenced type with regions %s", pprust::ty_to_str(ty)];
        if cx.region_is_relevant(r) {
            cx.add_rp(cx.item_id);
        }
      }

      _ => {}
    }

    // if this references another named type, add the dependency
    // to the dep_map.  If the type is not defined in this crate,
    // then check whether it is region-parameterized and consider
    // that as a direct dependency.
    alt ty.node {
      ast::ty_path(_, id) {
        alt cx.def_map.get(id) {
          ast::def_ty(did) | ast::def_class(did) {
            if did.crate == ast::local_crate {
                cx.add_dep(did.node, cx.item_id);
            } else {
                let cstore = cx.sess.cstore;
                if csearch::get_region_param(cstore, did) {
                    #debug["reference to external, rp'd type %s",
                           pprust::ty_to_str(ty)];
                    cx.add_rp(cx.item_id);
                }
            }
          }
          _ {}
        }
      }
      _ {}
    }

    visit::visit_ty(ty, cx, visitor);
}

fn determine_rp_in_crate(sess: session,
                         ast_map: ast_map::map,
                         def_map: resolve::def_map,
                         crate: @ast::crate) -> region_paramd_items {
    let cx = @{sess: sess,
               ast_map: ast_map,
               def_map: def_map,
               region_paramd_items: int_hash(),
               dep_map: int_hash(),
               worklist: dvec(),
               mut item_id: 0,
               mut anon_implies_rp: false};

    // gather up the base set, worklist and dep_map:
    let visitor = visit::mk_vt(@{
        visit_fn: determine_rp_in_fn,
        visit_item: determine_rp_in_item,
        visit_ty: determine_rp_in_ty,
        visit_ty_method: determine_rp_in_ty_method,
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, cx, visitor);

    // propagate indirect dependencies
    while cx.worklist.len() != 0 {
        let id = cx.worklist.pop();
        #debug["popped %d from worklist", id];
        alt cx.dep_map.find(id) {
          none {}
          some(vec) {
            for vec.each |to_id| {
                cx.add_rp(to_id);
            }
          }
        }
    }

    // return final set
    ret cx.region_paramd_items;
}