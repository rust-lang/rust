#[doc = "

# Borrow check

This pass is in job of enforcing *memory safety* and *purity*.  As
memory safety is by far the more complex topic, I'll focus on that in
this description, but purity will be covered later on. In the context
of Rust, memory safety means three basic things:

- no writes to immutable memory;
- all pointers point to non-freed memory;
- all pointers point to memory of the same type as the pointer.

The last point might seem confusing: after all, for the most part,
this condition is guaranteed by the type check.  However, there are
two cases where the type check effectively delegates to borrow check.

The first case has to do with enums.  If there is a pointer to the
interior of an enum, and the enum is in a mutable location (such as a
local variable or field declared to be mutable), it is possible that
the user will overwrite the enum with a new value of a different
variant, and thus effectively change the type of the memory that the
pointer is pointing at.

The second case has to do with mutability.  Basically, the type
checker has only a limited understanding of mutability.  It will allow
(for example) the user to get an immutable pointer with the address of
a mutable local variable.  It will also allow a `@mut T` or `~mut T`
pointer to be borrowed as a `&r.T` pointer.  These seeming oversights
are in fact intentional; they allow the user to temporarily treat a
mutable value as immutable.  It is up to the borrow check to guarantee
that the value in question is not in fact mutated during the lifetime
`r` of the reference.

# Summary of the safety check

In order to enforce mutability, the borrow check has three tricks up
its sleeve.

First, data which is uniquely tied to the current stack frame (that'll
be defined shortly) is tracked very precisely.  This means that, for
example, if an immutable pointer to a mutable local variable is
created, the borrowck will simply check for assignments to that
particular local variable: no other memory is affected.

Second, if the data is not uniquely tied to the stack frame, it may
still be possible to ensure its validity by rooting garbage collected
pointers at runtime.  For example, if there is a mutable local
variable `x` of type `@T`, and its contents are borrowed with an
expression like `&*x`, then the value of `x` will be rooted (today,
that means its ref count will be temporary increased) for the lifetime
of the reference that is created.  This means that the pointer remains
valid even if `x` is reassigned.

Finally, if neither of these two solutions are applicable, then we
require that all operations within the scope of the reference be
*pure*.  A pure operation is effectively one that does not write to
any aliasable memory.  This means that it is still possible to write
to local variables or other data that is uniquely tied to the stack
frame (there's that term again; formal definition still pending) but
not to data reached via a `&T` or `@T` pointer.  Such writes could
possibly have the side-effect of causing the data which must remain
valid to be overwritten.

# Possible future directions

There are numerous ways that the `borrowck` could be strengthened, but
these are the two most likely:

- flow-sensitivity: we do not currently consider flow at all but only
  block-scoping.  This means that innocent code like the following is
  rejected:

      let mut x: int;
      ...
      x = 5;
      let y: &int = &x; // immutable ptr created
      ...

  The reason is that the scope of the pointer `y` is the entire
  enclosing block, and the assignment `x = 5` occurs within that
  block.  The analysis is not smart enough to see that `x = 5` always
  happens before the immutable pointer is created.  This is relatively
  easy to fix and will surely be fixed at some point.

- finer-grained purity checks: currently, our fallback for
  guaranteeing random references into mutable, aliasable memory is to
  require *total purity*.  This is rather strong.  We could use local
  type-based alias analysis to distinguish writes that could not
  possibly invalid the references which must be guaranteed.  This
  would only work within the function boundaries; function calls would
  still require total purity.  This seems less likely to be
  implemented in the short term as it would make the code
  significantly more complex; there is currently no code to analyze
  the types and determine the possible impacts of a write.

# Terminology

A **loan** is .

# How the code works

The borrow check code is divided into several major modules, each of
which is documented in its own file.

The `gather_loans` and `check_loans` are the two major passes of the
analysis.  The `gather_loans` pass runs over the IR once to determine
what memory must remain valid and for how long.  Its name is a bit of
a misnomer; it does in fact gather up the set of loans which are
granted, but it also determines when @T pointers must be rooted and
for which scopes purity must be required.

The `check_loans` pass walks the IR and examines the loans and purity
requirements computed in `gather_loans`.  It checks to ensure that (a)
the conditions of all loans are honored; (b) no contradictory loans
were granted (for example, loaning out the same memory as mutable and
immutable simultaneously); and (c) any purity requirements are
honored.

The remaining modules are helper modules used by `gather_loans` and
`check_loans`:

- `categorization` has the job of analyzing an expression to determine
  what kind of memory is used in evaluating it (for example, where
  dereferences occur and what kind of pointer is dereferenced; whether
  the memory is mutable; etc)
- `loan` determines when data uniquely tied to the stack frame can be
  loaned out.
- `preserve` determines what actions (if any) must be taken to preserve
  aliasable data.  This is the code which decides when to root
  an @T pointer or to require purity.

# Maps that are created

Borrowck results in two maps.

- `root_map`: identifies those expressions or patterns whose result
  needs to be rooted.  Conceptually the root_map maps from an
  expression or pattern node to a `node_id` identifying the scope for
  which the expression must be rooted (this `node_id` should identify
  a block or call).  The actual key to the map is not an expression id,
  however, but a `root_map_key`, which combines an expression id with a
  deref count and is used to cope with auto-deref.

- `mutbl_map`: identifies those local variables which are modified or
  moved. This is used by trans to guarantee that such variables are
  given a memory location and not used as immediates.

"];

import syntax::ast;
import syntax::ast::{mutability, m_mutbl, m_imm, m_const};
import syntax::visit;
import syntax::ast_util;
import syntax::ast_map;
import syntax::codemap::span;
import util::ppaux::{ty_to_str, region_to_str};
import std::map::{int_hash, hashmap, set};
import std::list;
import std::list::{list, cons, nil};
import result::{result, ok, err, extensions};
import syntax::print::pprust;
import util::common::indenter;
import ast_util::op_expr_callee_id;
import ty::to_str;
import driver::session::session;
import dvec::{dvec, extensions};

export check_crate, root_map, mutbl_map;

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               last_use_map: liveness::last_use_map,
               crate: @ast::crate) -> (root_map, mutbl_map) {
    let bccx = @{tcx: tcx,
                 method_map: method_map,
                 last_use_map: last_use_map,
                 binding_map: int_hash(),
                 root_map: root_map(),
                 mutbl_map: int_hash()};

    let req_maps = gather_loans::gather_loans(bccx, crate);
    check_loans::check_loans(bccx, req_maps, crate);
    ret (bccx.root_map, bccx.mutbl_map);
}

// ----------------------------------------------------------------------
// Type definitions

type borrowck_ctxt = @{tcx: ty::ctxt,
                       method_map: typeck::method_map,
                       last_use_map: liveness::last_use_map,
                       binding_map: binding_map,
                       root_map: root_map,
                       mutbl_map: mutbl_map};

// a map mapping id's of expressions of gc'd type (@T, @[], etc) where
// the box needs to be kept live to the id of the scope for which they
// must stay live.
type root_map = hashmap<root_map_key, ast::node_id>;

// the keys to the root map combine the `id` of the expression with
// the number of types that it is autodereferenced.  So, for example,
// if you have an expression `x.f` and x has type ~@T, we could add an
// entry {id:x, derefs:0} to refer to `x` itself, `{id:x, derefs:1}`
// to refer to the deref of the unique pointer, and so on.
type root_map_key = {id: ast::node_id, derefs: uint};

// set of ids of local vars / formal arguments that are modified / moved.
// this is used in trans for optimization purposes.
type mutbl_map = std::map::hashmap<ast::node_id, ()>;

// maps from each binding's id to the mutability of the location it
// points at.  See gather_loan.rs for more detail (search for binding_map)
type binding_map = std::map::hashmap<ast::node_id, ast::mutability>;

// Errors that can occur"]
enum bckerr_code {
    err_mut_uniq,
    err_mut_variant,
    err_preserve_gc,
    err_mutbl(ast::mutability,
              ast::mutability)
}

// Combination of an error code and the categorization of the expression
// that caused it
type bckerr = {cmt: cmt, code: bckerr_code};

// shorthand for something that fails with `bckerr` or succeeds with `T`
type bckres<T> = result<T, bckerr>;

enum categorization {
    cat_rvalue,                     // result of eval'ing some misc expr
    cat_special(special_kind),      //
    cat_local(ast::node_id),        // local variable
    cat_binding(ast::node_id),      // pattern binding
    cat_arg(ast::node_id),          // formal argument
    cat_stack_upvar(cmt),           // upvar in stack closure
    cat_deref(cmt, uint, ptr_kind), // deref of a ptr
    cat_comp(cmt, comp_kind),       // adjust to locate an internal component
    cat_discr(cmt, ast::node_id),   // alt discriminant (see preserve())
}

// different kinds of pointers:
enum ptr_kind {uniq_ptr, gc_ptr, region_ptr, unsafe_ptr}

// I am coining the term "components" to mean "pieces of a data
// structure accessible without a dereference":
enum comp_kind {
    comp_tuple,                  // elt in a tuple
    comp_variant(ast::def_id),   // internals to a variant of given enum
    comp_field(ast::ident,       // name of field
               ast::mutability), // declared mutability of field
    comp_index(ty::t,            // type of vec/str/etc being deref'd
               ast::mutability)  // mutability of vec content
}

// We pun on *T to mean both actual deref of a ptr as well
// as accessing of components:
enum deref_kind {deref_ptr(ptr_kind), deref_comp(comp_kind)}

// different kinds of expressions we might evaluate
enum special_kind {
    sk_method,
    sk_static_item,
    sk_self,
    sk_heap_upvar
}

// a complete categorization of a value indicating where it originated
// and how it is located, as well as the mutability of the memory in
// which the value is stored.
type cmt = @{id: ast::node_id,        // id of expr/pat producing this value
             span: span,              // span of same expr/pat
             cat: categorization,     // categorization of expr
             lp: option<@loan_path>,  // loan path for expr, if any
             mutbl: ast::mutability,  // mutability of expr as lvalue
             ty: ty::t};              // type of the expr

// a loan path is like a category, but it exists only when the data is
// interior to the stack frame.  loan paths are used as the key to a
// map indicating what is borrowed at any point in time.
enum loan_path {
    lp_local(ast::node_id),
    lp_arg(ast::node_id),
    lp_deref(@loan_path, ptr_kind),
    lp_comp(@loan_path, comp_kind)
}

// a complete record of a loan that was granted
type loan = {lp: @loan_path, cmt: cmt, mutbl: ast::mutability};

// maps computed by `gather_loans` that are then used by `check_loans`
type req_maps = {
    req_loan_map: hashmap<ast::node_id, @dvec<@dvec<loan>>>,
    pure_map: hashmap<ast::node_id, bckerr>
};

fn save_and_restore<T:copy,U>(&save_and_restore_t: T, f: fn() -> U) -> U {
    let old_save_and_restore_t = save_and_restore_t;
    let u <- f();
    save_and_restore_t = old_save_and_restore_t;
    ret u;
}

#[doc = "Creates and returns a new root_map"]
fn root_map() -> root_map {
    ret hashmap(root_map_key_hash, root_map_key_eq);

    fn root_map_key_eq(k1: root_map_key, k2: root_map_key) -> bool {
        k1.id == k2.id && k1.derefs == k2.derefs
    }

    fn root_map_key_hash(k: root_map_key) -> uint {
        (k.id << 4) as uint | k.derefs
    }
}

// ___________________________________________________________________________
// Misc

iface ast_node {
    fn id() -> ast::node_id;
    fn span() -> span;
}

impl of ast_node for @ast::expr {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

impl of ast_node for @ast::pat {
    fn id() -> ast::node_id { self.id }
    fn span() -> span { self.span }
}

impl methods for ty::ctxt {
    fn ty<N: ast_node>(node: N) -> ty::t {
        ty::node_id_to_type(self, node.id())
    }
}

impl error_methods for borrowck_ctxt {
    fn report_if_err(bres: bckres<()>) {
        alt bres {
          ok(()) { }
          err(e) { self.report(e); }
        }
    }

    fn report(err: bckerr) {
        self.span_err(
            err.cmt.span,
            #fmt["illegal borrow: %s",
                 self.bckerr_code_to_str(err.code)]);
    }

    fn span_err(s: span, m: str) {
        self.tcx.sess.span_err(s, m);
    }

    fn span_note(s: span, m: str) {
        self.tcx.sess.span_note(s, m);
    }

    fn add_to_mutbl_map(cmt: cmt) {
        alt cmt.cat {
          cat_local(id) | cat_arg(id) {
            self.mutbl_map.insert(id, ());
          }
          cat_stack_upvar(cmt) {
            self.add_to_mutbl_map(cmt);
          }
          _ {}
        }
    }
}

impl to_str_methods for borrowck_ctxt {
    fn cat_to_repr(cat: categorization) -> str {
        alt cat {
          cat_special(sk_method) { "method" }
          cat_special(sk_static_item) { "static_item" }
          cat_special(sk_self) { "self" }
          cat_special(sk_heap_upvar) { "heap-upvar" }
          cat_stack_upvar(_) { "stack-upvar" }
          cat_rvalue { "rvalue" }
          cat_local(node_id) { #fmt["local(%d)", node_id] }
          cat_binding(node_id) { #fmt["binding(%d)", node_id] }
          cat_arg(node_id) { #fmt["arg(%d)", node_id] }
          cat_deref(cmt, derefs, ptr) {
            #fmt["%s->(%s, %u)", self.cat_to_repr(cmt.cat),
                 self.ptr_sigil(ptr), derefs]
          }
          cat_comp(cmt, comp) {
            #fmt["%s.%s", self.cat_to_repr(cmt.cat), self.comp_to_repr(comp)]
          }
          cat_discr(cmt, _) { self.cat_to_repr(cmt.cat) }
        }
    }

    fn mut_to_str(mutbl: ast::mutability) -> str {
        alt mutbl {
          m_mutbl { "mutable" }
          m_const { "const" }
          m_imm { "immutable" }
        }
    }

    fn ptr_sigil(ptr: ptr_kind) -> str {
        alt ptr {
          uniq_ptr { "~" }
          gc_ptr { "@" }
          region_ptr { "&" }
          unsafe_ptr { "*" }
        }
    }

    fn comp_to_repr(comp: comp_kind) -> str {
        alt comp {
          comp_field(fld, _) { *fld }
          comp_index(*) { "[]" }
          comp_tuple { "()" }
          comp_variant(_) { "<enum>" }
        }
    }

    fn lp_to_str(lp: @loan_path) -> str {
        alt *lp {
          lp_local(node_id) {
            #fmt["local(%d)", node_id]
          }
          lp_arg(node_id) {
            #fmt["arg(%d)", node_id]
          }
          lp_deref(lp, ptr) {
            #fmt["%s->(%s)", self.lp_to_str(lp),
                 self.ptr_sigil(ptr)]
          }
          lp_comp(lp, comp) {
            #fmt["%s.%s", self.lp_to_str(lp),
                 self.comp_to_repr(comp)]
          }
        }
    }

    fn cmt_to_repr(cmt: cmt) -> str {
        #fmt["{%s id:%d m:%s lp:%s ty:%s}",
             self.cat_to_repr(cmt.cat),
             cmt.id,
             self.mut_to_str(cmt.mutbl),
             cmt.lp.map_default("none", { |p| self.lp_to_str(p) }),
             ty_to_str(self.tcx, cmt.ty)]
    }

    fn pk_to_sigil(pk: ptr_kind) -> str {
        alt pk {
          uniq_ptr {"~"}
          gc_ptr {"@"}
          region_ptr {"&"}
          unsafe_ptr {"*"}
        }
    }

    fn cmt_to_str(cmt: cmt) -> str {
        let mut_str = self.mut_to_str(cmt.mutbl);
        alt cmt.cat {
          cat_special(sk_method) { "method" }
          cat_special(sk_static_item) { "static item" }
          cat_special(sk_self) { "self reference" }
          cat_special(sk_heap_upvar) { "variable declared in an outer block" }
          cat_rvalue { "non-lvalue" }
          cat_local(_) { mut_str + " local variable" }
          cat_binding(_) { "pattern binding" }
          cat_arg(_) { "argument" }
          cat_deref(_, _, pk) { #fmt["dereference of %s %s pointer",
                                     mut_str, self.pk_to_sigil(pk)] }
          cat_stack_upvar(_) {
            mut_str + " variable declared in an outer block"
          }
          cat_comp(_, comp_field(*)) { mut_str + " field" }
          cat_comp(_, comp_tuple) { "tuple content" }
          cat_comp(_, comp_variant(_)) { "enum content" }
          cat_comp(_, comp_index(t, _)) {
            alt ty::get(t).struct {
              ty::ty_vec(*) | ty::ty_evec(*) {
                mut_str + " vec content"
              }

              ty::ty_str | ty::ty_estr(*) {
                mut_str + " str content"
              }

              _ { mut_str + " indexed content" }
            }
          }
          cat_discr(cmt, _) {
            self.cmt_to_str(cmt)
          }
        }
    }

    fn bckerr_code_to_str(code: bckerr_code) -> str {
        alt code {
          err_mutbl(req, act) {
            #fmt["creating %s alias to aliasable, %s memory",
                 self.mut_to_str(req), self.mut_to_str(act)]
          }
          err_mut_uniq {
            "unique value in aliasable, mutable location"
          }
          err_mut_variant {
            "enum variant in aliasable, mutable location"
          }
          err_preserve_gc {
            "GC'd value would have to be preserved for longer \
                 than the scope of the function"
          }
        }
    }
}

// The inherent mutability of a component is its default mutability
// assuming it is embedded in an immutable context.  In general, the
// mutability can be "overridden" if the component is embedded in a
// mutable structure.
fn inherent_mutability(ck: comp_kind) -> mutability {
    alt ck {
      comp_tuple | comp_variant(_)        {m_imm}
      comp_field(_, m) | comp_index(_, m) {m}
    }
}
