/*!
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

# Definition of unstable memory

The primary danger to safety arises due to *unstable memory*.
Unstable memory is memory whose validity or type may change as a
result of an assignment, move, or a variable going out of scope.
There are two cases in Rust where memory is unstable: the contents of
unique boxes and enums.

Unique boxes are unstable because when the variable containing the
unique box is re-assigned, moves, or goes out of scope, the unique box
is freed or---in the case of a move---potentially given to another
task.  In either case, if there is an extant and usable pointer into
the box, then safety guarantees would be compromised.

Enum values are unstable because they are reassigned the types of
their contents may change if they are assigned with a different
variant than they had previously.

# Safety criteria that must be enforced

Whenever a piece of memory is borrowed for lifetime L, there are two
things which the borrow checker must guarantee.  First, it must
guarantee that the memory address will remain allocated (and owned by
the current task) for the entirety of the lifetime L.  Second, it must
guarantee that the type of the data will not change for the entirety
of the lifetime L.  In exchange, the region-based type system will
guarantee that the pointer is not used outside the lifetime L.  These
guarantees are to some extent independent but are also inter-related.

In some cases, the type of a pointer cannot be invalidated but the
lifetime can.  For example, imagine a pointer to the interior of
a shared box like:

    let mut x = @mut {f: 5, g: 6};
    let y = &mut x.f;

Here, a pointer was created to the interior of a shared box which
contains a record.  Even if `*x` were to be mutated like so:

    *x = {f: 6, g: 7};

This would cause `*y` to change from 5 to 6, but the pointer pointer
`y` remains valid.  It still points at an integer even if that integer
has been overwritten.

However, if we were to reassign `x` itself, like so:

    x = @{f: 6, g: 7};

This could potentially invalidate `y`, because if `x` were the final
reference to the shared box, then that memory would be released and
now `y` points at freed memory.  (We will see that to prevent this
scenario we will *root* shared boxes that reside in mutable memory
whose contents are borrowed; rooting means that we create a temporary
to ensure that the box is not collected).

In other cases, like an enum on the stack, the memory cannot be freed
but its type can change:

    let mut x = some(5);
    alt x {
      some(ref y) => { ... }
      none => { ... }
    }

Here as before, the pointer `y` would be invalidated if we were to
reassign `x` to `none`.  (We will see that this case is prevented
because borrowck tracks data which resides on the stack and prevents
variables from reassigned if there may be pointers to their interior)

Finally, in some cases, both dangers can arise.  For example, something
like the following:

    let mut x = ~some(5);
    alt x {
      ~some(ref y) => { ... }
      ~none => { ... }
    }

In this case, if `x` to be reassigned or `*x` were to be mutated, then
the pointer `y` would be invalided.  (This case is also prevented by
borrowck tracking data which is owned by the current stack frame)

# Summary of the safety check

In order to enforce mutability, the borrow check has a few tricks up
its sleeve:

- When data is owned by the current stack frame, we can identify every
  possible assignment to a local variable and simply prevent
  potentially dangerous assignments directly.

- If data is owned by a shared box, we can root the box to increase
  its lifetime.

- If data is found within a borrowed pointer, we can assume that the
  data will remain live for the entirety of the borrowed pointer.

- We can rely on the fact that pure actions (such as calling pure
  functions) do not mutate data which is not owned by the current
  stack frame.

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
 */

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
import ty::to_str;
import driver::session::session;
import dvec::{dvec, extensions};

export check_crate, root_map, mutbl_map;

fn check_crate(tcx: ty::ctxt,
               method_map: typeck::method_map,
               last_use_map: liveness::last_use_map,
               crate: @ast::crate) -> (root_map, mutbl_map) {

    let bccx = borrowck_ctxt_(@{tcx: tcx,
                                method_map: method_map,
                                last_use_map: last_use_map,
                                binding_map: int_hash(),
                                root_map: root_map(),
                                mutbl_map: int_hash(),
                                mut loaned_paths_same: 0,
                                mut loaned_paths_imm: 0,
                                mut stable_paths: 0,
                                mut req_pure_paths: 0,
                                mut guaranteed_paths: 0});

    let req_maps = gather_loans::gather_loans(bccx, crate);
    check_loans::check_loans(bccx, req_maps, crate);

    if tcx.sess.borrowck_stats() {
        io::println(~"--- borrowck stats ---");
        io::println(fmt!{"paths requiring guarantees: %u",
                        bccx.guaranteed_paths});
        io::println(fmt!{"paths requiring loans     : %s",
                         make_stat(bccx, bccx.loaned_paths_same)});
        io::println(fmt!{"paths requiring imm loans : %s",
                         make_stat(bccx, bccx.loaned_paths_imm)});
        io::println(fmt!{"stable paths              : %s",
                         make_stat(bccx, bccx.stable_paths)});
        io::println(fmt!{"paths requiring purity    : %s",
                         make_stat(bccx, bccx.req_pure_paths)});
    }

    ret (bccx.root_map, bccx.mutbl_map);

    fn make_stat(bccx: borrowck_ctxt, stat: uint) -> ~str {
        let stat_f = stat as float;
        let total = bccx.guaranteed_paths as float;
        fmt!{"%u (%.0f%%)", stat  , stat_f * 100f / total}
    }
}

// ----------------------------------------------------------------------
// Type definitions

type borrowck_ctxt_ = {tcx: ty::ctxt,
                       method_map: typeck::method_map,
                       last_use_map: liveness::last_use_map,
                       binding_map: binding_map,
                       root_map: root_map,
                       mutbl_map: mutbl_map,

                       // Statistics:
                       mut loaned_paths_same: uint,
                       mut loaned_paths_imm: uint,
                       mut stable_paths: uint,
                       mut req_pure_paths: uint,
                       mut guaranteed_paths: uint};

enum borrowck_ctxt {
    borrowck_ctxt_(@borrowck_ctxt_)
}

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
    err_root_not_permitted,
    err_mutbl(ast::mutability, ast::mutability),
    err_out_of_root_scope(ty::region, ty::region), // superscope, subscope
    err_out_of_scope(ty::region, ty::region) // superscope, subscope
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
enum ptr_kind {uniq_ptr, gc_ptr, region_ptr(ty::region), unsafe_ptr}

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

/// a complete record of a loan that was granted
type loan = {lp: @loan_path, cmt: cmt, mutbl: ast::mutability};

/// maps computed by `gather_loans` that are then used by `check_loans`
///
/// - `req_loan_map`: map from each block/expr to the required loans needed
///   for the duration of that block/expr
/// - `pure_map`: map from block/expr that must be pure to the error message
///   that should be reported if they are not pure
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

/// Creates and returns a new root_map
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

trait get_type_for_node {
    fn ty<N: ast_node>(node: N) -> ty::t;
}

impl methods of get_type_for_node for ty::ctxt {
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
            fmt!{"illegal borrow: %s",
                 self.bckerr_code_to_str(err.code)});
    }

    fn span_err(s: span, m: ~str) {
        self.tcx.sess.span_err(s, m);
    }

    fn span_note(s: span, m: ~str) {
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
    fn cat_to_repr(cat: categorization) -> ~str {
        alt cat {
          cat_special(sk_method) { ~"method" }
          cat_special(sk_static_item) { ~"static_item" }
          cat_special(sk_self) { ~"self" }
          cat_special(sk_heap_upvar) { ~"heap-upvar" }
          cat_stack_upvar(_) { ~"stack-upvar" }
          cat_rvalue { ~"rvalue" }
          cat_local(node_id) { fmt!{"local(%d)", node_id} }
          cat_binding(node_id) { fmt!{"binding(%d)", node_id} }
          cat_arg(node_id) { fmt!{"arg(%d)", node_id} }
          cat_deref(cmt, derefs, ptr) {
            fmt!{"%s->(%s, %u)", self.cat_to_repr(cmt.cat),
                 self.ptr_sigil(ptr), derefs}
          }
          cat_comp(cmt, comp) {
            fmt!{"%s.%s", self.cat_to_repr(cmt.cat), self.comp_to_repr(comp)}
          }
          cat_discr(cmt, _) { self.cat_to_repr(cmt.cat) }
        }
    }

    fn mut_to_str(mutbl: ast::mutability) -> ~str {
        alt mutbl {
          m_mutbl { ~"mutable" }
          m_const { ~"const" }
          m_imm { ~"immutable" }
        }
    }

    fn ptr_sigil(ptr: ptr_kind) -> ~str {
        alt ptr {
          uniq_ptr { ~"~" }
          gc_ptr { ~"@" }
          region_ptr(_) { ~"&" }
          unsafe_ptr { ~"*" }
        }
    }

    fn comp_to_repr(comp: comp_kind) -> ~str {
        alt comp {
          comp_field(fld, _) { *fld }
          comp_index(*) { ~"[]" }
          comp_tuple { ~"()" }
          comp_variant(_) { ~"<enum>" }
        }
    }

    fn lp_to_str(lp: @loan_path) -> ~str {
        alt *lp {
          lp_local(node_id) {
            fmt!{"local(%d)", node_id}
          }
          lp_arg(node_id) {
            fmt!{"arg(%d)", node_id}
          }
          lp_deref(lp, ptr) {
            fmt!{"%s->(%s)", self.lp_to_str(lp),
                 self.ptr_sigil(ptr)}
          }
          lp_comp(lp, comp) {
            fmt!{"%s.%s", self.lp_to_str(lp),
                 self.comp_to_repr(comp)}
          }
        }
    }

    fn cmt_to_repr(cmt: cmt) -> ~str {
        fmt!{"{%s id:%d m:%s lp:%s ty:%s}",
             self.cat_to_repr(cmt.cat),
             cmt.id,
             self.mut_to_str(cmt.mutbl),
             cmt.lp.map_default(~"none", |p| self.lp_to_str(p) ),
             ty_to_str(self.tcx, cmt.ty)}
    }

    fn cmt_to_str(cmt: cmt) -> ~str {
        let mut_str = self.mut_to_str(cmt.mutbl);
        alt cmt.cat {
          cat_special(sk_method) { ~"method" }
          cat_special(sk_static_item) { ~"static item" }
          cat_special(sk_self) { ~"self reference" }
          cat_special(sk_heap_upvar) {
              ~"captured outer variable in a heap closure"
          }
          cat_rvalue { ~"non-lvalue" }
          cat_local(_) { mut_str + ~" local variable" }
          cat_binding(_) { ~"pattern binding" }
          cat_arg(_) { ~"argument" }
          cat_deref(_, _, pk) { fmt!{"dereference of %s %s pointer",
                                     mut_str, self.ptr_sigil(pk)} }
          cat_stack_upvar(_) {
            ~"captured outer " + mut_str + ~" variable in a stack closure"
          }
          cat_comp(_, comp_field(*)) { mut_str + ~" field" }
          cat_comp(_, comp_tuple) { ~"tuple content" }
          cat_comp(_, comp_variant(_)) { ~"enum content" }
          cat_comp(_, comp_index(t, _)) {
            alt ty::get(t).struct {
              ty::ty_evec(*) {
                mut_str + ~" vec content"
              }

              ty::ty_estr(*) {
                mut_str + ~" str content"
              }

              _ { mut_str + ~" indexed content" }
            }
          }
          cat_discr(cmt, _) {
            self.cmt_to_str(cmt)
          }
        }
    }

    fn bckerr_code_to_str(code: bckerr_code) -> ~str {
        alt code {
          err_mutbl(req, act) {
            fmt!{"creating %s alias to aliasable, %s memory",
                 self.mut_to_str(req), self.mut_to_str(act)}
          }
          err_mut_uniq {
            ~"unique value in aliasable, mutable location"
          }
          err_mut_variant {
            ~"enum variant in aliasable, mutable location"
          }
          err_root_not_permitted {
            // note: I don't expect users to ever see this error
            // message, reasons are discussed in attempt_root() in
            // preserve.rs.
            ~"rooting is not permitted"
          }
          err_out_of_root_scope(super_scope, sub_scope) {
            fmt!{"managed value would have to be rooted for lifetime %s, \
                  but can only be rooted for lifetime %s",
                 self.region_to_str(sub_scope),
                 self.region_to_str(super_scope)}
          }
          err_out_of_scope(super_scope, sub_scope) {
            fmt!{"borrowed pointer has lifetime %s, \
                  but the borrowed value only has lifetime %s",
                 self.region_to_str(sub_scope),
                 self.region_to_str(super_scope)}
          }
        }
    }

    fn region_to_str(r: ty::region) -> ~str {
        region_to_str(self.tcx, r)
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
