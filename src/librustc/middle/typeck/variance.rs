// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

This file infers the variance of type and lifetime parameters. The
algorithm is taken from Section 4 of the paper "Taming the Wildcards:
Combining Definition- and Use-Site Variance" published in PLDI'11 and
written by Altidor et al., and hereafter referred to as The Paper.

This inference is explicitly designed *not* to consider the uses of
types within code. To determine the variance of type parameters
defined on type `X`, we only consider the definition of the type `X`
and the definitions of any types it references.

We only infer variance for type parameters found on *types*: structs,
enums, and traits. We do not infer variance for type parameters found
on fns or impls. This is because those things are not type definitions
and variance doesn't really make sense in that context.

It is worth covering what variance means in each case. For structs and
enums, I think it is fairly straightforward. The variance of the type
or lifetime parameters defines whether `T<A>` is a subtype of `T<B>`
(resp. `T<'a>` and `T<'b>`) based on the relationship of `A` and `B`
(resp. `'a` and `'b`). (FIXME #3598 -- we do not currently make use of
the variances we compute for type parameters.)

### Variance on traits

The meaning of variance for trait parameters is more subtle and worth
expanding upon. There are in fact two uses of the variance values we
compute.

#### Trait variance and object types

The first is for object types. Just as with structs and enums, we can
decide the subtyping relationship between two object types `&Trait<A>`
and `&Trait<B>` based on the relationship of `A` and `B`. Note that
for object types we ignore the `Self` type parameter -- it is unknown,
and the nature of dynamic dispatch ensures that we will always call a
function that is expected the appropriate `Self` type. However, we
must be careful with the other type parameters, or else we could end
up calling a function that is expecting one type but provided another.

To see what I mean, consider a trait like so:

    trait ConvertTo<A> {
        fn convertTo(&self) -> A;
    }

Intuitively, If we had one object `O=&ConvertTo<Object>` and another
`S=&ConvertTo<String>`, then `S <: O` because `String <: Object`
(presuming Java-like "string" and "object" types, my go to examples
for subtyping). The actual algorithm would be to compare the
(explicit) type parameters pairwise respecting their variance: here,
the type parameter A is covariant (it appears only in a return
position), and hence we require that `String <: Object`.

You'll note though that we did not consider the binding for the
(implicit) `Self` type parameter: in fact, it is unknown, so that's
good. The reason we can ignore that parameter is precisely because we
don't need to know its value until a call occurs, and at that time (as
you said) the dynamic nature of virtual dispatch means the code we run
will be correct for whatever value `Self` happens to be bound to for
the particular object whose method we called. `Self` is thus different
from `A`, because the caller requires that `A` be known in order to
know the return type of the method `convertTo()`. (As an aside, we
have rules preventing methods where `Self` appears outside of the
receiver position from being called via an object.)

#### Trait variance and vtable resolution

But traits aren't only used with objects. They're also used when
deciding whether a given impl satisfies a given trait bound (or should
be -- FIXME #5781). To set the scene here, imagine I had a function:

    fn convertAll<A,T:ConvertTo<A>>(v: &[T]) {
        ...
    }

Now imagine that I have an implementation of `ConvertTo` for `Object`:

    impl ConvertTo<int> for Object { ... }

And I want to call `convertAll` on an array of strings. Suppose
further that for whatever reason I specifically supply the value of
`String` for the type parameter `T`:

    let mut vector = ~["string", ...];
    convertAll::<int, String>(v);

Is this legal? To put another way, can we apply the `impl` for
`Object` to the type `String`? The answer is yes, but to see why
we have to expand out what will happen:

- `convertAll` will create a pointer to one of the entries in the
  vector, which will have type `&String`
- It will then call the impl of `convertTo()` that is intended
  for use with objects. This has the type:

      fn(self: &Object) -> int

  It is ok to provide a value for `self` of type `&String` because
  `&String <: &Object`.

OK, so intuitively we want this to be legal, so let's bring this back
to variance and see whether we are computing the correct result. We
must first figure out how to phrase the question "is an impl for
`Object,int` usable where an impl for `String,int` is expected?"

Maybe it's helpful to think of a dictionary-passing implementation of
type classes. In that case, `convertAll()` takes an implicit parameter
representing the impl. In short, we *have* an impl of type:

    V_O = ConvertTo<int> for Object

and the function prototype expects an impl of type:

    V_S = ConvertTo<int> for String

As with any argument, this is legal if the type of the value given
(`V_O`) is a subtype of the type expected (`V_S`). So is `V_O <: V_S`?
The answer will depend on the variance of the various parameters. In
this case, because the `Self` parameter is contravariant and `A` is
covariant, it means that:

    V_O <: V_S iff
        int <: int
        String <: Object

These conditions are satisfied and so we are happy.

### The algorithm

The basic idea is quite straightforward. We iterate over the types
defined and, for each use of a type parameter X, accumulate a
constraint indicating that the variance of X must be valid for the
variance of that use site. We then iteratively refine the variance of
X until all constraints are met. There is *always* a sol'n, because at
the limit we can declare all type parameters to be invariant and all
constraints will be satisfied.

As a simple example, consider:

    enum Option<A> { Some(A), None }
    enum OptionalFn<B> { Some(|B|), None }
    enum OptionalMap<C> { Some(|C| -> C), None }

Here, we will generate the constraints:

    1. V(A) <= +
    2. V(B) <= -
    3. V(C) <= +
    4. V(C) <= -

These indicate that (1) the variance of A must be at most covariant;
(2) the variance of B must be at most contravariant; and (3, 4) the
variance of C must be at most covariant *and* contravariant. All of these
results are based on a variance lattice defined as follows:

      *      Top (bivariant)
   -     +
      o      Bottom (invariant)

Based on this lattice, the solution V(A)=+, V(B)=-, V(C)=o is the
optimal solution. Note that there is always a naive solution which
just declares all variables to be invariant.

You may be wondering why fixed-point iteration is required. The reason
is that the variance of a use site may itself be a function of the
variance of other type parameters. In full generality, our constraints
take the form:

    V(X) <= Term
    Term := + | - | * | o | V(X) | Term x Term

Here the notation V(X) indicates the variance of a type/region
parameter `X` with respect to its defining class. `Term x Term`
represents the "variance transform" as defined in the paper:

  If the variance of a type variable `X` in type expression `E` is `V2`
  and the definition-site variance of the [corresponding] type parameter
  of a class `C` is `V1`, then the variance of `X` in the type expression
  `C<E>` is `V3 = V1.xform(V2)`.

*/

use collections::HashMap;
use arena;
use arena::Arena;
use middle::ty;
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util;
use syntax::owned_slice::OwnedSlice;
use syntax::visit;
use syntax::visit::Visitor;
use util::ppaux::Repr;

pub fn infer_variance(tcx: &ty::ctxt,
                      krate: &ast::Crate) {
    let mut arena = arena::Arena::new();
    let terms_cx = determine_parameters_to_be_inferred(tcx, &mut arena, krate);
    let constraints_cx = add_constraints_from_crate(terms_cx, krate);
    solve_constraints(constraints_cx);
}

/**************************************************************************
 * Representing terms
 *
 * Terms are structured as a straightforward tree. Rather than rely on
 * GC, we allocate terms out of a bounded arena (the lifetime of this
 * arena is the lifetime 'a that is threaded around).
 *
 * We assign a unique index to each type/region parameter whose variance
 * is to be inferred. We refer to such variables as "inferreds". An
 * `InferredIndex` is a newtype'd int representing the index of such
 * a variable.
 */

type VarianceTermPtr<'a> = &'a VarianceTerm<'a>;

struct InferredIndex(uint);

enum VarianceTerm<'a> {
    ConstantTerm(ty::Variance),
    TransformTerm(VarianceTermPtr<'a>, VarianceTermPtr<'a>),
    InferredTerm(InferredIndex),
}

impl<'a> fmt::Show for VarianceTerm<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConstantTerm(c1) => write!(f, "{}", c1),
            TransformTerm(v1, v2) => write!(f, "({} \u00D7 {})", v1, v2),
            InferredTerm(id) => write!(f, "[{}]", { let InferredIndex(i) = id; i })
        }
    }
}

/**************************************************************************
 * The first pass over the crate simply builds up the set of inferreds.
 */

struct TermsContext<'a> {
    tcx: &'a ty::ctxt,
    arena: &'a Arena,

    empty_variances: Rc<ty::ItemVariances>,

    // Maps from the node id of a type/generic parameter to the
    // corresponding inferred index.
    inferred_map: HashMap<ast::NodeId, InferredIndex>,

    // Maps from an InferredIndex to the info for that variable.
    inferred_infos: Vec<InferredInfo<'a>> ,
}

enum ParamKind { TypeParam, RegionParam, SelfParam }

struct InferredInfo<'a> {
    item_id: ast::NodeId,
    kind: ParamKind,
    index: uint,
    param_id: ast::NodeId,
    term: VarianceTermPtr<'a>,
}

fn determine_parameters_to_be_inferred<'a>(tcx: &'a ty::ctxt,
                                           arena: &'a mut Arena,
                                           krate: &ast::Crate)
                                           -> TermsContext<'a> {
    let mut terms_cx = TermsContext {
        tcx: tcx,
        arena: arena,
        inferred_map: HashMap::new(),
        inferred_infos: Vec::new(),

        // cache and share the variance struct used for items with
        // no type/region parameters
        empty_variances: Rc::new(ty::ItemVariances {
            self_param: None,
            type_params: OwnedSlice::empty(),
            region_params: OwnedSlice::empty()
        })
    };

    visit::walk_crate(&mut terms_cx, krate, ());

    terms_cx
}

impl<'a> TermsContext<'a> {
    fn add_inferred(&mut self,
                    item_id: ast::NodeId,
                    kind: ParamKind,
                    index: uint,
                    param_id: ast::NodeId) {
        let inf_index = InferredIndex(self.inferred_infos.len());
        let term = self.arena.alloc(|| InferredTerm(inf_index));
        self.inferred_infos.push(InferredInfo { item_id: item_id,
                                                kind: kind,
                                                index: index,
                                                param_id: param_id,
                                                term: term });
        let newly_added = self.inferred_map.insert(param_id, inf_index);
        assert!(newly_added);

        debug!("add_inferred(item_id={}, \
                kind={:?}, \
                index={}, \
                param_id={},
                inf_index={:?})",
                item_id, kind, index, param_id, inf_index);
    }

    fn num_inferred(&self) -> uint {
        self.inferred_infos.len()
    }
}

impl<'a> Visitor<()> for TermsContext<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        debug!("add_inferreds for item {}", item.repr(self.tcx));

        let inferreds_on_entry = self.num_inferred();

        // NB: In the code below for writing the results back into the
        // tcx, we rely on the fact that all inferreds for a particular
        // item are assigned continuous indices.
        match item.node {
            ast::ItemTrait(..) => {
                self.add_inferred(item.id, SelfParam, 0, item.id);
            }
            _ => { }
        }

        match item.node {
            ast::ItemEnum(_, ref generics) |
            ast::ItemStruct(_, ref generics) |
            ast::ItemTrait(ref generics, _, _, _) => {
                for (i, p) in generics.lifetimes.iter().enumerate() {
                    self.add_inferred(item.id, RegionParam, i, p.id);
                }
                for (i, p) in generics.ty_params.iter().enumerate() {
                    self.add_inferred(item.id, TypeParam, i, p.id);
                }

                // If this item has no type or lifetime parameters,
                // then there are no variances to infer, so just
                // insert an empty entry into the variance map.
                // Arguably we could just leave the map empty in this
                // case but it seems cleaner to be able to distinguish
                // "invalid item id" from "item id with no
                // parameters".
                if self.num_inferred() == inferreds_on_entry {
                    let newly_added = self.tcx.item_variance_map.borrow_mut().insert(
                        ast_util::local_def(item.id),
                        self.empty_variances.clone());
                    assert!(newly_added);
                }

                visit::walk_item(self, item, ());
            }

            ast::ItemImpl(..) |
            ast::ItemStatic(..) |
            ast::ItemFn(..) |
            ast::ItemMod(..) |
            ast::ItemForeignMod(..) |
            ast::ItemTy(..) |
            ast::ItemMac(..) => {
                visit::walk_item(self, item, ());
            }
        }
    }
}

/**************************************************************************
 * Constraint construction and representation
 *
 * The second pass over the AST determines the set of constraints.
 * We walk the set of items and, for each member, generate new constraints.
 */

struct ConstraintContext<'a> {
    terms_cx: TermsContext<'a>,

    // These are the def-id of the std::kinds::marker::InvariantType,
    // std::kinds::marker::InvariantLifetime, and so on. The arrays
    // are indexed by the `ParamKind` (type, lifetime, self). Note
    // that there are no marker types for self, so the entries for
    // self are always None.
    invariant_lang_items: [Option<ast::DefId>, ..3],
    covariant_lang_items: [Option<ast::DefId>, ..3],
    contravariant_lang_items: [Option<ast::DefId>, ..3],

    // These are pointers to common `ConstantTerm` instances
    covariant: VarianceTermPtr<'a>,
    contravariant: VarianceTermPtr<'a>,
    invariant: VarianceTermPtr<'a>,
    bivariant: VarianceTermPtr<'a>,

    constraints: Vec<Constraint<'a>> ,
}

/// Declares that the variable `decl_id` appears in a location with
/// variance `variance`.
struct Constraint<'a> {
    inferred: InferredIndex,
    variance: &'a VarianceTerm<'a>,
}

fn add_constraints_from_crate<'a>(terms_cx: TermsContext<'a>,
                                  krate: &ast::Crate)
                                  -> ConstraintContext<'a> {
    let mut invariant_lang_items = [None, ..3];
    let mut covariant_lang_items = [None, ..3];
    let mut contravariant_lang_items = [None, ..3];

    covariant_lang_items[TypeParam as uint] =
        terms_cx.tcx.lang_items.covariant_type();
    covariant_lang_items[RegionParam as uint] =
        terms_cx.tcx.lang_items.covariant_lifetime();

    contravariant_lang_items[TypeParam as uint] =
        terms_cx.tcx.lang_items.contravariant_type();
    contravariant_lang_items[RegionParam as uint] =
        terms_cx.tcx.lang_items.contravariant_lifetime();

    invariant_lang_items[TypeParam as uint] =
        terms_cx.tcx.lang_items.invariant_type();
    invariant_lang_items[RegionParam as uint] =
        terms_cx.tcx.lang_items.invariant_lifetime();

    let covariant = terms_cx.arena.alloc(|| ConstantTerm(ty::Covariant));
    let contravariant = terms_cx.arena.alloc(|| ConstantTerm(ty::Contravariant));
    let invariant = terms_cx.arena.alloc(|| ConstantTerm(ty::Invariant));
    let bivariant = terms_cx.arena.alloc(|| ConstantTerm(ty::Bivariant));
    let mut constraint_cx = ConstraintContext {
        terms_cx: terms_cx,

        invariant_lang_items: invariant_lang_items,
        covariant_lang_items: covariant_lang_items,
        contravariant_lang_items: contravariant_lang_items,

        covariant: covariant,
        contravariant: contravariant,
        invariant: invariant,
        bivariant: bivariant,
        constraints: Vec::new(),
    };
    visit::walk_crate(&mut constraint_cx, krate, ());
    constraint_cx
}

impl<'a> Visitor<()> for ConstraintContext<'a> {
    fn visit_item(&mut self, item: &ast::Item, _: ()) {
        let did = ast_util::local_def(item.id);
        let tcx = self.terms_cx.tcx;

        match item.node {
            ast::ItemEnum(ref enum_definition, _) => {
                // Hack: If we directly call `ty::enum_variants`, it
                // annoyingly takes it upon itself to run off and
                // evaluate the discriminants eagerly (*grumpy* that's
                // not the typical pattern). This results in double
                // error messages because typeck goes off and does
                // this at a later time. All we really care about is
                // the types of the variant arguments, so we just call
                // `ty::VariantInfo::from_ast_variant()` ourselves
                // here, mainly so as to mask the differences between
                // struct-like enums and so forth.
                for &ast_variant in enum_definition.variants.iter() {
                    let variant =
                        ty::VariantInfo::from_ast_variant(tcx,
                                                          ast_variant,
                                                          /*discrimant*/ 0);
                    for &arg_ty in variant.args.iter() {
                        self.add_constraints_from_ty(arg_ty, self.covariant);
                    }
                }
            }

            ast::ItemStruct(..) => {
                let struct_fields = ty::lookup_struct_fields(tcx, did);
                for field_info in struct_fields.iter() {
                    assert_eq!(field_info.id.krate, ast::LOCAL_CRATE);
                    let field_ty = ty::node_id_to_type(tcx, field_info.id.node);
                    self.add_constraints_from_ty(field_ty, self.covariant);
                }
            }

            ast::ItemTrait(..) => {
                let methods = ty::trait_methods(tcx, did);
                for method in methods.iter() {
                    self.add_constraints_from_sig(
                        &method.fty.sig, self.covariant);
                }
            }

            ast::ItemStatic(..) |
            ast::ItemFn(..) |
            ast::ItemMod(..) |
            ast::ItemForeignMod(..) |
            ast::ItemTy(..) |
            ast::ItemImpl(..) |
            ast::ItemMac(..) => {
                visit::walk_item(self, item, ());
            }
        }
    }
}

/// Is `param_id` a lifetime according to `map`?
fn is_lifetime(map: &ast_map::Map, param_id: ast::NodeId) -> bool {
    match map.find(param_id) {
        Some(ast_map::NodeLifetime(..)) => true, _ => false
    }
}

impl<'a> ConstraintContext<'a> {
    fn tcx(&self) -> &'a ty::ctxt {
        self.terms_cx.tcx
    }

    fn inferred_index(&self, param_id: ast::NodeId) -> InferredIndex {
        match self.terms_cx.inferred_map.find(&param_id) {
            Some(&index) => index,
            None => {
                self.tcx().sess.bug(format!(
                        "no inferred index entry for {}",
                        self.tcx().map.node_to_str(param_id)).as_slice());
            }
        }
    }

    fn find_binding_for_lifetime(&self, param_id: ast::NodeId) -> ast::NodeId {
        let tcx = self.terms_cx.tcx;
        assert!(is_lifetime(&tcx.map, param_id));
        match tcx.named_region_map.find(&param_id) {
            Some(&ast::DefEarlyBoundRegion(_, lifetime_decl_id))
                => lifetime_decl_id,
            Some(_) => fail!("should not encounter non early-bound cases"),

            // The lookup should only fail when `param_id` is
            // itself a lifetime binding: use it as the decl_id.
            None    => param_id,
        }

    }

    /// Is `param_id` a type parameter for which we infer variance?
    fn is_to_be_inferred(&self, param_id: ast::NodeId) -> bool {
        let result = self.terms_cx.inferred_map.contains_key(&param_id);

        // To safe-guard against invalid inferred_map constructions,
        // double-check if variance is inferred at some use of a type
        // parameter (by inspecting parent of its binding declaration
        // to see if it is introduced by a type or by a fn/impl).

        let check_result = |this:&ConstraintContext| -> bool {
            let tcx = this.terms_cx.tcx;
            let decl_id = this.find_binding_for_lifetime(param_id);
            // Currently only called on lifetimes; double-checking that.
            assert!(is_lifetime(&tcx.map, param_id));
            let parent_id = tcx.map.get_parent(decl_id);
            let parent = tcx.map.find(parent_id).unwrap_or_else(
                || fail!("tcx.map missing entry for id: {}", parent_id));

            let is_inferred;
            macro_rules! cannot_happen { () => { {
                fail!("invalid parent: {:s} for {:s}",
                      tcx.map.node_to_str(parent_id),
                      tcx.map.node_to_str(param_id));
            } } }

            match parent {
                ast_map::NodeItem(p) => {
                    match p.node {
                        ast::ItemTy(..) |
                        ast::ItemEnum(..) |
                        ast::ItemStruct(..) |
                        ast::ItemTrait(..)   => is_inferred = true,
                        ast::ItemFn(..)      => is_inferred = false,
                        _                    => cannot_happen!(),
                    }
                }
                ast_map::NodeTraitMethod(..) => is_inferred = false,
                ast_map::NodeMethod(_)       => is_inferred = false,
                _                            => cannot_happen!(),
            }

            return is_inferred;
        };

        assert_eq!(result, check_result(self));

        return result;
    }

    fn declared_variance(&self,
                         param_def_id: ast::DefId,
                         item_def_id: ast::DefId,
                         kind: ParamKind,
                         index: uint)
                         -> VarianceTermPtr<'a> {
        /*!
         * Returns a variance term representing the declared variance of
         * the type/region parameter with the given id.
         */

        assert_eq!(param_def_id.krate, item_def_id.krate);

        if self.invariant_lang_items[kind as uint] == Some(item_def_id) {
            self.invariant
        } else if self.covariant_lang_items[kind as uint] == Some(item_def_id) {
            self.covariant
        } else if self.contravariant_lang_items[kind as uint] == Some(item_def_id) {
            self.contravariant
        } else if param_def_id.krate == ast::LOCAL_CRATE {
            // Parameter on an item defined within current crate:
            // variance not yet inferred, so return a symbolic
            // variance.
            let InferredIndex(index) = self.inferred_index(param_def_id.node);
            self.terms_cx.inferred_infos.get(index).term
        } else {
            // Parameter on an item defined within another crate:
            // variance already inferred, just look it up.
            let variances = ty::item_variances(self.tcx(), item_def_id);
            let variance = match kind {
                SelfParam => variances.self_param.unwrap(),
                TypeParam => *variances.type_params.get(index),
                RegionParam => *variances.region_params.get(index),
            };
            self.constant_term(variance)
        }
    }

    fn add_constraint(&mut self,
                      InferredIndex(index): InferredIndex,
                      variance: VarianceTermPtr<'a>) {
        debug!("add_constraint(index={}, variance={})",
                index, variance.to_str());
        self.constraints.push(Constraint { inferred: InferredIndex(index),
                                           variance: variance });
    }

    fn contravariant(&mut self,
                     variance: VarianceTermPtr<'a>)
                     -> VarianceTermPtr<'a> {
        self.xform(variance, self.contravariant)
    }

    fn invariant(&mut self,
                 variance: VarianceTermPtr<'a>)
                 -> VarianceTermPtr<'a> {
        self.xform(variance, self.invariant)
    }

    fn constant_term(&self, v: ty::Variance) -> VarianceTermPtr<'a> {
        match v {
            ty::Covariant => self.covariant,
            ty::Invariant => self.invariant,
            ty::Contravariant => self.contravariant,
            ty::Bivariant => self.bivariant,
        }
    }

    fn xform(&mut self,
             v1: VarianceTermPtr<'a>,
             v2: VarianceTermPtr<'a>)
             -> VarianceTermPtr<'a> {
        match (*v1, *v2) {
            (_, ConstantTerm(ty::Covariant)) => {
                // Applying a "covariant" transform is always a no-op
                v1
            }

            (ConstantTerm(c1), ConstantTerm(c2)) => {
                self.constant_term(c1.xform(c2))
            }

            _ => {
                self.terms_cx.arena.alloc(|| TransformTerm(v1, v2))
            }
        }
    }

    /// Adds constraints appropriate for an instance of `ty` appearing
    /// in a context with ambient variance `variance`
    fn add_constraints_from_ty(&mut self,
                               ty: ty::t,
                               variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_ty(ty={})", ty.repr(self.tcx()));

        match ty::get(ty).sty {
            ty::ty_nil | ty::ty_bot | ty::ty_bool |
            ty::ty_char | ty::ty_int(_) | ty::ty_uint(_) |
            ty::ty_float(_) | ty::ty_str => {
                /* leaf type -- noop */
            }

            ty::ty_rptr(region, ref mt) => {
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(region, contra);
                self.add_constraints_from_mt(mt, variance);
            }

            ty::ty_vec(ref mt, _) => {
                self.add_constraints_from_mt(mt, variance);
            }

            ty::ty_uniq(typ) | ty::ty_box(typ) => {
                self.add_constraints_from_ty(typ, variance);
            }

            ty::ty_ptr(ref mt) => {
                self.add_constraints_from_mt(mt, variance);
            }

            ty::ty_tup(ref subtys) => {
                for &subty in subtys.iter() {
                    self.add_constraints_from_ty(subty, variance);
                }
            }

            ty::ty_enum(def_id, ref substs) |
            ty::ty_struct(def_id, ref substs) => {
                let item_type = ty::lookup_item_type(self.tcx(), def_id);
                self.add_constraints_from_substs(def_id, &item_type.generics,
                                                 substs, variance);
            }

            ty::ty_trait(box ty::TyTrait { def_id, ref substs, .. }) => {
                let trait_def = ty::lookup_trait_def(self.tcx(), def_id);
                self.add_constraints_from_substs(def_id, &trait_def.generics,
                                                 substs, variance);
            }

            ty::ty_param(ty::param_ty { def_id: ref def_id, .. }) => {
                assert_eq!(def_id.krate, ast::LOCAL_CRATE);
                match self.terms_cx.inferred_map.find(&def_id.node) {
                    Some(&index) => {
                        self.add_constraint(index, variance);
                    }
                    None => {
                        // We do not infer variance for type parameters
                        // declared on methods. They will not be present
                        // in the inferred_map.
                    }
                }
            }

            ty::ty_self(ref def_id) => {
                assert_eq!(def_id.krate, ast::LOCAL_CRATE);
                let index = self.inferred_index(def_id.node);
                self.add_constraint(index, variance);
            }

            ty::ty_bare_fn(ty::BareFnTy { ref sig, .. }) |
            ty::ty_closure(box ty::ClosureTy {
                    ref sig,
                    store: ty::UniqTraitStore,
                    ..
                }) => {
                self.add_constraints_from_sig(sig, variance);
            }

            ty::ty_closure(box ty::ClosureTy { ref sig,
                    store: ty::RegionTraitStore(region, _), .. }) => {
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(region, contra);
                self.add_constraints_from_sig(sig, variance);
            }

            ty::ty_infer(..) | ty::ty_err => {
                self.tcx().sess.bug(
                    format!("unexpected type encountered in \
                            variance inference: {}",
                            ty.repr(self.tcx())).as_slice());
            }
        }
    }


    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_substs(&mut self,
                                   def_id: ast::DefId,
                                   generics: &ty::Generics,
                                   substs: &ty::substs,
                                   variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_substs(def_id={:?})", def_id);

        for (i, p) in generics.type_param_defs().iter().enumerate() {
            let variance_decl =
                self.declared_variance(p.def_id, def_id, TypeParam, i);
            let variance_i = self.xform(variance, variance_decl);
            self.add_constraints_from_ty(*substs.tps.get(i), variance_i);
        }

        match substs.regions {
            ty::ErasedRegions => {}
            ty::NonerasedRegions(ref rps) => {
                for (i, p) in generics.region_param_defs().iter().enumerate() {
                    let variance_decl =
                        self.declared_variance(p.def_id, def_id, RegionParam, i);
                    let variance_i = self.xform(variance, variance_decl);
                    self.add_constraints_from_region(*rps.get(i), variance_i);
                }
            }
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig(&mut self,
                                sig: &ty::FnSig,
                                variance: VarianceTermPtr<'a>) {
        let contra = self.contravariant(variance);
        for &input in sig.inputs.iter() {
            self.add_constraints_from_ty(input, contra);
        }
        self.add_constraints_from_ty(sig.output, variance);
    }

    /// Adds constraints appropriate for a region appearing in a
    /// context with ambient variance `variance`
    fn add_constraints_from_region(&mut self,
                                   region: ty::Region,
                                   variance: VarianceTermPtr<'a>) {
        match region {
            ty::ReEarlyBound(param_id, _, _) => {
                if self.is_to_be_inferred(param_id) {
                    let index = self.inferred_index(param_id);
                    self.add_constraint(index, variance);
                }
            }

            ty::ReStatic => { }

            ty::ReLateBound(..) => {
                // We do not infer variance for region parameters on
                // methods or in fn types.
            }

            ty::ReFree(..) | ty::ReScope(..) | ty::ReInfer(..) |
            ty::ReEmpty => {
                // We don't expect to see anything but 'static or bound
                // regions when visiting member types or method types.
                self.tcx()
                    .sess
                    .bug(format!("unexpected region encountered in variance \
                                  inference: {}",
                                 region.repr(self.tcx())).as_slice());
            }
        }
    }

    /// Adds constraints appropriate for a mutability-type pair
    /// appearing in a context with ambient variance `variance`
    fn add_constraints_from_mt(&mut self,
                               mt: &ty::mt,
                               variance: VarianceTermPtr<'a>) {
        match mt.mutbl {
            ast::MutMutable => {
                let invar = self.invariant(variance);
                self.add_constraints_from_ty(mt.ty, invar);
            }

            ast::MutImmutable => {
                self.add_constraints_from_ty(mt.ty, variance);
            }
        }
    }
}

/**************************************************************************
 * Constraint solving
 *
 * The final phase iterates over the constraints, refining the variance
 * for each inferred until a fixed point is reached. This will be the
 * optimal solution to the constraints. The final variance for each
 * inferred is then written into the `variance_map` in the tcx.
 */

struct SolveContext<'a> {
    terms_cx: TermsContext<'a>,
    constraints: Vec<Constraint<'a>> ,

    // Maps from an InferredIndex to the inferred value for that variable.
    solutions: Vec<ty::Variance> }

fn solve_constraints(constraints_cx: ConstraintContext) {
    let ConstraintContext { terms_cx, constraints, .. } = constraints_cx;
    let solutions = Vec::from_elem(terms_cx.num_inferred(), ty::Bivariant);
    let mut solutions_cx = SolveContext {
        terms_cx: terms_cx,
        constraints: constraints,
        solutions: solutions
    };
    solutions_cx.solve();
    solutions_cx.write();
}

impl<'a> SolveContext<'a> {
    fn solve(&mut self) {
        // Propagate constraints until a fixed point is reached.  Note
        // that the maximum number of iterations is 2C where C is the
        // number of constraints (each variable can change values at most
        // twice). Since number of constraints is linear in size of the
        // input, so is the inference process.
        let mut changed = true;
        while changed {
            changed = false;

            for constraint in self.constraints.iter() {
                let Constraint { inferred, variance: term } = *constraint;
                let InferredIndex(inferred) = inferred;
                let variance = self.evaluate(term);
                let old_value = *self.solutions.get(inferred);
                let new_value = glb(variance, old_value);
                if old_value != new_value {
                    debug!("Updating inferred {} (node {}) \
                            from {:?} to {:?} due to {}",
                            inferred,
                            self.terms_cx
                                .inferred_infos
                                .get(inferred)
                                .param_id,
                            old_value,
                            new_value,
                            term.to_str());

                    *self.solutions.get_mut(inferred) = new_value;
                    changed = true;
                }
            }
        }
    }

    fn write(&self) {
        // Collect all the variances for a particular item and stick
        // them into the variance map. We rely on the fact that we
        // generate all the inferreds for a particular item
        // consecutively (that is, we collect solutions for an item
        // until we see a new item id, and we assume (1) the solutions
        // are in the same order as the type parameters were declared
        // and (2) all solutions or a given item appear before a new
        // item id).

        let tcx = self.terms_cx.tcx;
        let solutions = &self.solutions;
        let inferred_infos = &self.terms_cx.inferred_infos;
        let mut index = 0;
        let num_inferred = self.terms_cx.num_inferred();
        while index < num_inferred {
            let item_id = inferred_infos.get(index).item_id;
            let mut self_param = None;
            let mut type_params = vec!();
            let mut region_params = vec!();

            while index < num_inferred &&
                  inferred_infos.get(index).item_id == item_id {
                let info = inferred_infos.get(index);
                match info.kind {
                    SelfParam => {
                        assert!(self_param.is_none());
                        self_param = Some(*solutions.get(index));
                    }
                    TypeParam => {
                        type_params.push(*solutions.get(index));
                    }
                    RegionParam => {
                        region_params.push(*solutions.get(index));
                    }
                }
                index += 1;
            }

            let item_variances = ty::ItemVariances {
                self_param: self_param,
                type_params: OwnedSlice::from_vec(type_params),
                region_params: OwnedSlice::from_vec(region_params)
            };
            debug!("item_id={} item_variances={}",
                    item_id,
                    item_variances.repr(tcx));

            let item_def_id = ast_util::local_def(item_id);

            // For unit testing: check for a special "rustc_variance"
            // attribute and report an error with various results if found.
            if ty::has_attr(tcx, item_def_id, "rustc_variance") {
                let found = item_variances.repr(tcx);
                tcx.sess.span_err(tcx.map.span(item_id), found.as_slice());
            }

            let newly_added = tcx.item_variance_map.borrow_mut()
                                 .insert(item_def_id, Rc::new(item_variances));
            assert!(newly_added);
        }
    }

    fn evaluate(&self, term: VarianceTermPtr<'a>) -> ty::Variance {
        match *term {
            ConstantTerm(v) => {
                v
            }

            TransformTerm(t1, t2) => {
                let v1 = self.evaluate(t1);
                let v2 = self.evaluate(t2);
                v1.xform(v2)
            }

            InferredTerm(InferredIndex(index)) => {
                *self.solutions.get(index)
            }
        }
    }
}

/**************************************************************************
 * Miscellany transformations on variance
 */

trait Xform {
    fn xform(self, v: Self) -> Self;
}

impl Xform for ty::Variance {
    fn xform(self, v: ty::Variance) -> ty::Variance {
        // "Variance transformation", Figure 1 of The Paper
        match (self, v) {
            // Figure 1, column 1.
            (ty::Covariant, ty::Covariant) => ty::Covariant,
            (ty::Covariant, ty::Contravariant) => ty::Contravariant,
            (ty::Covariant, ty::Invariant) => ty::Invariant,
            (ty::Covariant, ty::Bivariant) => ty::Bivariant,

            // Figure 1, column 2.
            (ty::Contravariant, ty::Covariant) => ty::Contravariant,
            (ty::Contravariant, ty::Contravariant) => ty::Covariant,
            (ty::Contravariant, ty::Invariant) => ty::Invariant,
            (ty::Contravariant, ty::Bivariant) => ty::Bivariant,

            // Figure 1, column 3.
            (ty::Invariant, _) => ty::Invariant,

            // Figure 1, column 4.
            (ty::Bivariant, _) => ty::Bivariant,
        }
    }
}

fn glb(v1: ty::Variance, v2: ty::Variance) -> ty::Variance {
    // Greatest lower bound of the variance lattice as
    // defined in The Paper:
    //
    //       *
    //    -     +
    //       o
    match (v1, v2) {
        (ty::Invariant, _) | (_, ty::Invariant) => ty::Invariant,

        (ty::Covariant, ty::Contravariant) => ty::Invariant,
        (ty::Contravariant, ty::Covariant) => ty::Invariant,

        (ty::Covariant, ty::Covariant) => ty::Covariant,

        (ty::Contravariant, ty::Contravariant) => ty::Contravariant,

        (x, ty::Bivariant) | (ty::Bivariant, x) => x,
    }
}
