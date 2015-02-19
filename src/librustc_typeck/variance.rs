// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This file infers the variance of type and lifetime parameters. The
//! algorithm is taken from Section 4 of the paper "Taming the Wildcards:
//! Combining Definition- and Use-Site Variance" published in PLDI'11 and
//! written by Altidor et al., and hereafter referred to as The Paper.
//!
//! This inference is explicitly designed *not* to consider the uses of
//! types within code. To determine the variance of type parameters
//! defined on type `X`, we only consider the definition of the type `X`
//! and the definitions of any types it references.
//!
//! We only infer variance for type parameters found on *types*: structs,
//! enums, and traits. We do not infer variance for type parameters found
//! on fns or impls. This is because those things are not type definitions
//! and variance doesn't really make sense in that context.
//!
//! It is worth covering what variance means in each case. For structs and
//! enums, I think it is fairly straightforward. The variance of the type
//! or lifetime parameters defines whether `T<A>` is a subtype of `T<B>`
//! (resp. `T<'a>` and `T<'b>`) based on the relationship of `A` and `B`
//! (resp. `'a` and `'b`). (FIXME #3598 -- we do not currently make use of
//! the variances we compute for type parameters.)
//!
//! ### Variance on traits
//!
//! The meaning of variance for trait parameters is more subtle and worth
//! expanding upon. There are in fact two uses of the variance values we
//! compute.
//!
//! #### Trait variance and object types
//!
//! The first is for object types. Just as with structs and enums, we can
//! decide the subtyping relationship between two object types `&Trait<A>`
//! and `&Trait<B>` based on the relationship of `A` and `B`. Note that
//! for object types we ignore the `Self` type parameter -- it is unknown,
//! and the nature of dynamic dispatch ensures that we will always call a
//! function that is expected the appropriate `Self` type. However, we
//! must be careful with the other type parameters, or else we could end
//! up calling a function that is expecting one type but provided another.
//!
//! To see what I mean, consider a trait like so:
//!
//!     trait ConvertTo<A> {
//!         fn convertTo(&self) -> A;
//!     }
//!
//! Intuitively, If we had one object `O=&ConvertTo<Object>` and another
//! `S=&ConvertTo<String>`, then `S <: O` because `String <: Object`
//! (presuming Java-like "string" and "object" types, my go to examples
//! for subtyping). The actual algorithm would be to compare the
//! (explicit) type parameters pairwise respecting their variance: here,
//! the type parameter A is covariant (it appears only in a return
//! position), and hence we require that `String <: Object`.
//!
//! You'll note though that we did not consider the binding for the
//! (implicit) `Self` type parameter: in fact, it is unknown, so that's
//! good. The reason we can ignore that parameter is precisely because we
//! don't need to know its value until a call occurs, and at that time (as
//! you said) the dynamic nature of virtual dispatch means the code we run
//! will be correct for whatever value `Self` happens to be bound to for
//! the particular object whose method we called. `Self` is thus different
//! from `A`, because the caller requires that `A` be known in order to
//! know the return type of the method `convertTo()`. (As an aside, we
//! have rules preventing methods where `Self` appears outside of the
//! receiver position from being called via an object.)
//!
//! #### Trait variance and vtable resolution
//!
//! But traits aren't only used with objects. They're also used when
//! deciding whether a given impl satisfies a given trait bound. To set the
//! scene here, imagine I had a function:
//!
//!     fn convertAll<A,T:ConvertTo<A>>(v: &[T]) {
//!         ...
//!     }
//!
//! Now imagine that I have an implementation of `ConvertTo` for `Object`:
//!
//!     impl ConvertTo<int> for Object { ... }
//!
//! And I want to call `convertAll` on an array of strings. Suppose
//! further that for whatever reason I specifically supply the value of
//! `String` for the type parameter `T`:
//!
//!     let mut vector = ~["string", ...];
//!     convertAll::<int, String>(v);
//!
//! Is this legal? To put another way, can we apply the `impl` for
//! `Object` to the type `String`? The answer is yes, but to see why
//! we have to expand out what will happen:
//!
//! - `convertAll` will create a pointer to one of the entries in the
//!   vector, which will have type `&String`
//! - It will then call the impl of `convertTo()` that is intended
//!   for use with objects. This has the type:
//!
//!       fn(self: &Object) -> int
//!
//!   It is ok to provide a value for `self` of type `&String` because
//!   `&String <: &Object`.
//!
//! OK, so intuitively we want this to be legal, so let's bring this back
//! to variance and see whether we are computing the correct result. We
//! must first figure out how to phrase the question "is an impl for
//! `Object,int` usable where an impl for `String,int` is expected?"
//!
//! Maybe it's helpful to think of a dictionary-passing implementation of
//! type classes. In that case, `convertAll()` takes an implicit parameter
//! representing the impl. In short, we *have* an impl of type:
//!
//!     V_O = ConvertTo<int> for Object
//!
//! and the function prototype expects an impl of type:
//!
//!     V_S = ConvertTo<int> for String
//!
//! As with any argument, this is legal if the type of the value given
//! (`V_O`) is a subtype of the type expected (`V_S`). So is `V_O <: V_S`?
//! The answer will depend on the variance of the various parameters. In
//! this case, because the `Self` parameter is contravariant and `A` is
//! covariant, it means that:
//!
//!     V_O <: V_S iff
//!         int <: int
//!         String <: Object
//!
//! These conditions are satisfied and so we are happy.
//!
//! ### The algorithm
//!
//! The basic idea is quite straightforward. We iterate over the types
//! defined and, for each use of a type parameter X, accumulate a
//! constraint indicating that the variance of X must be valid for the
//! variance of that use site. We then iteratively refine the variance of
//! X until all constraints are met. There is *always* a sol'n, because at
//! the limit we can declare all type parameters to be invariant and all
//! constraints will be satisfied.
//!
//! As a simple example, consider:
//!
//!     enum Option<A> { Some(A), None }
//!     enum OptionalFn<B> { Some(|B|), None }
//!     enum OptionalMap<C> { Some(|C| -> C), None }
//!
//! Here, we will generate the constraints:
//!
//!     1. V(A) <= +
//!     2. V(B) <= -
//!     3. V(C) <= +
//!     4. V(C) <= -
//!
//! These indicate that (1) the variance of A must be at most covariant;
//! (2) the variance of B must be at most contravariant; and (3, 4) the
//! variance of C must be at most covariant *and* contravariant. All of these
//! results are based on a variance lattice defined as follows:
//!
//!       *      Top (bivariant)
//!    -     +
//!       o      Bottom (invariant)
//!
//! Based on this lattice, the solution V(A)=+, V(B)=-, V(C)=o is the
//! optimal solution. Note that there is always a naive solution which
//! just declares all variables to be invariant.
//!
//! You may be wondering why fixed-point iteration is required. The reason
//! is that the variance of a use site may itself be a function of the
//! variance of other type parameters. In full generality, our constraints
//! take the form:
//!
//!     V(X) <= Term
//!     Term := + | - | * | o | V(X) | Term x Term
//!
//! Here the notation V(X) indicates the variance of a type/region
//! parameter `X` with respect to its defining class. `Term x Term`
//! represents the "variance transform" as defined in the paper:
//!
//!   If the variance of a type variable `X` in type expression `E` is `V2`
//!   and the definition-site variance of the [corresponding] type parameter
//!   of a class `C` is `V1`, then the variance of `X` in the type expression
//!   `C<E>` is `V3 = V1.xform(V2)`.
//!
//! ### Constraints
//!
//! If I have a struct or enum with where clauses:
//!
//!     struct Foo<T:Bar> { ... }
//!
//! you might wonder whether the variance of `T` with respect to `Bar`
//! affects the variance `T` with respect to `Foo`. I claim no.  The
//! reason: assume that `T` is invariant w/r/t `Bar` but covariant w/r/t
//! `Foo`. And then we have a `Foo<X>` that is upcast to `Foo<Y>`, where
//! `X <: Y`. However, while `X : Bar`, `Y : Bar` does not hold.  In that
//! case, the upcast will be illegal, but not because of a variance
//! failure, but rather because the target type `Foo<Y>` is itself just
//! not well-formed. Basically we get to assume well-formedness of all
//! types involved before considering variance.

use self::VarianceTerm::*;
use self::ParamKind::*;

use arena;
use arena::TypedArena;
use middle::resolve_lifetime as rl;
use middle::subst;
use middle::subst::{ParamSpace, FnSpace, TypeSpace, SelfSpace, VecPerParamSpace};
use middle::ty::{self, Ty};
use std::fmt;
use std::rc::Rc;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util;
use syntax::visit;
use syntax::visit::Visitor;
use util::nodemap::NodeMap;
use util::ppaux::Repr;

pub fn infer_variance(tcx: &ty::ctxt) {
    let krate = tcx.map.krate();
    let mut arena = arena::TypedArena::new();
    let terms_cx = determine_parameters_to_be_inferred(tcx, &mut arena, krate);
    let constraints_cx = add_constraints_from_crate(terms_cx, krate);
    solve_constraints(constraints_cx);
    tcx.variance_computed.set(true);
}

// Representing terms
//
// Terms are structured as a straightforward tree. Rather than rely on
// GC, we allocate terms out of a bounded arena (the lifetime of this
// arena is the lifetime 'a that is threaded around).
//
// We assign a unique index to each type/region parameter whose variance
// is to be inferred. We refer to such variables as "inferreds". An
// `InferredIndex` is a newtype'd int representing the index of such
// a variable.

type VarianceTermPtr<'a> = &'a VarianceTerm<'a>;

#[derive(Copy, Debug)]
struct InferredIndex(uint);

#[derive(Copy)]
enum VarianceTerm<'a> {
    ConstantTerm(ty::Variance),
    TransformTerm(VarianceTermPtr<'a>, VarianceTermPtr<'a>),
    InferredTerm(InferredIndex),
}

impl<'a> fmt::Debug for VarianceTerm<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ConstantTerm(c1) => write!(f, "{:?}", c1),
            TransformTerm(v1, v2) => write!(f, "({:?} \u{00D7} {:?})", v1, v2),
            InferredTerm(id) => write!(f, "[{}]", { let InferredIndex(i) = id; i })
        }
    }
}

// The first pass over the crate simply builds up the set of inferreds.

struct TermsContext<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
    arena: &'a TypedArena<VarianceTerm<'a>>,

    empty_variances: Rc<ty::ItemVariances>,

    // For marker types, UnsafeCell, and other lang items where
    // variance is hardcoded, records the item-id and the hardcoded
    // variance.
    lang_items: Vec<(ast::NodeId, Vec<ty::Variance>)>,

    // Maps from the node id of a type/generic parameter to the
    // corresponding inferred index.
    inferred_map: NodeMap<InferredIndex>,

    // Maps from an InferredIndex to the info for that variable.
    inferred_infos: Vec<InferredInfo<'a>> ,
}

#[derive(Copy, Debug, PartialEq)]
enum ParamKind {
    TypeParam,
    RegionParam,
}

struct InferredInfo<'a> {
    item_id: ast::NodeId,
    kind: ParamKind,
    space: ParamSpace,
    index: uint,
    param_id: ast::NodeId,
    term: VarianceTermPtr<'a>,

    // Initial value to use for this parameter when inferring
    // variance. For most parameters, this is Bivariant. But for lang
    // items and input type parameters on traits, it is different.
    initial_variance: ty::Variance,
}

fn determine_parameters_to_be_inferred<'a, 'tcx>(tcx: &'a ty::ctxt<'tcx>,
                                                 arena: &'a mut TypedArena<VarianceTerm<'a>>,
                                                 krate: &ast::Crate)
                                                 -> TermsContext<'a, 'tcx> {
    let mut terms_cx = TermsContext {
        tcx: tcx,
        arena: arena,
        inferred_map: NodeMap(),
        inferred_infos: Vec::new(),

        lang_items: lang_items(tcx),

        // cache and share the variance struct used for items with
        // no type/region parameters
        empty_variances: Rc::new(ty::ItemVariances {
            types: VecPerParamSpace::empty(),
            regions: VecPerParamSpace::empty()
        })
    };

    visit::walk_crate(&mut terms_cx, krate);

    terms_cx
}

fn lang_items(tcx: &ty::ctxt) -> Vec<(ast::NodeId,Vec<ty::Variance>)> {
    let all = vec![
        (tcx.lang_items.phantom_fn(), vec![ty::Contravariant, ty::Covariant]),
        (tcx.lang_items.phantom_data(), vec![ty::Covariant]),
        (tcx.lang_items.unsafe_cell_type(), vec![ty::Invariant]),

        // Deprecated:
        (tcx.lang_items.covariant_type(), vec![ty::Covariant]),
        (tcx.lang_items.contravariant_type(), vec![ty::Contravariant]),
        (tcx.lang_items.invariant_type(), vec![ty::Invariant]),
        (tcx.lang_items.covariant_lifetime(), vec![ty::Covariant]),
        (tcx.lang_items.contravariant_lifetime(), vec![ty::Contravariant]),
        (tcx.lang_items.invariant_lifetime(), vec![ty::Invariant]),

        ];

    all.into_iter()
       .filter(|&(ref d,_)| d.is_some())
       .filter(|&(ref d,_)| d.as_ref().unwrap().krate == ast::LOCAL_CRATE)
       .map(|(d, v)| (d.unwrap().node, v))
       .collect()
}

impl<'a, 'tcx> TermsContext<'a, 'tcx> {
    fn add_inferreds_for_item(&mut self,
                              item_id: ast::NodeId,
                              has_self: bool,
                              generics: &ast::Generics)
    {
        /*!
         * Add "inferreds" for the generic parameters declared on this
         * item. This has a lot of annoying parameters because we are
         * trying to drive this from the AST, rather than the
         * ty::Generics, so that we can get span info -- but this
         * means we must accommodate syntactic distinctions.
         */

        // NB: In the code below for writing the results back into the
        // tcx, we rely on the fact that all inferreds for a particular
        // item are assigned continuous indices.

        let inferreds_on_entry = self.num_inferred();

        if has_self {
            self.add_inferred(item_id, TypeParam, SelfSpace, 0, item_id);
        }

        for (i, p) in generics.lifetimes.iter().enumerate() {
            let id = p.lifetime.id;
            self.add_inferred(item_id, RegionParam, TypeSpace, i, id);
        }

        for (i, p) in generics.ty_params.iter().enumerate() {
            self.add_inferred(item_id, TypeParam, TypeSpace, i, p.id);
        }

        // If this item has no type or lifetime parameters,
        // then there are no variances to infer, so just
        // insert an empty entry into the variance map.
        // Arguably we could just leave the map empty in this
        // case but it seems cleaner to be able to distinguish
        // "invalid item id" from "item id with no
        // parameters".
        if self.num_inferred() == inferreds_on_entry {
            let newly_added =
                self.tcx.item_variance_map.borrow_mut().insert(
                    ast_util::local_def(item_id),
                    self.empty_variances.clone()).is_none();
            assert!(newly_added);
        }
    }

    fn add_inferred(&mut self,
                    item_id: ast::NodeId,
                    kind: ParamKind,
                    space: ParamSpace,
                    index: uint,
                    param_id: ast::NodeId) {
        let inf_index = InferredIndex(self.inferred_infos.len());
        let term = self.arena.alloc(InferredTerm(inf_index));
        let initial_variance = self.pick_initial_variance(item_id, space, index);
        self.inferred_infos.push(InferredInfo { item_id: item_id,
                                                kind: kind,
                                                space: space,
                                                index: index,
                                                param_id: param_id,
                                                term: term,
                                                initial_variance: initial_variance });
        let newly_added = self.inferred_map.insert(param_id, inf_index).is_none();
        assert!(newly_added);

        debug!("add_inferred(item_path={}, \
                item_id={}, \
                kind={:?}, \
                space={:?}, \
                index={}, \
                param_id={}, \
                inf_index={:?}, \
                initial_variance={:?})",
               ty::item_path_str(self.tcx, ast_util::local_def(item_id)),
               item_id, kind, space, index, param_id, inf_index,
               initial_variance);
    }

    fn pick_initial_variance(&self,
                             item_id: ast::NodeId,
                             space: ParamSpace,
                             index: uint)
                             -> ty::Variance
    {
        match space {
            SelfSpace | FnSpace => {
                ty::Bivariant
            }

            TypeSpace => {
                match self.lang_items.iter().find(|&&(n, _)| n == item_id) {
                    Some(&(_, ref variances)) => variances[index],
                    None => ty::Bivariant
                }
            }
        }
    }

    fn num_inferred(&self) -> uint {
        self.inferred_infos.len()
    }
}

impl<'a, 'tcx, 'v> Visitor<'v> for TermsContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        debug!("add_inferreds for item {}", item.repr(self.tcx));

        match item.node {
            ast::ItemEnum(_, ref generics) |
            ast::ItemStruct(_, ref generics) => {
                self.add_inferreds_for_item(item.id, false, generics);
            }
            ast::ItemTrait(_, ref generics, _, _) => {
                self.add_inferreds_for_item(item.id, true, generics);
                visit::walk_item(self, item);
            }

            ast::ItemExternCrate(_) |
            ast::ItemUse(_) |
            ast::ItemImpl(..) |
            ast::ItemStatic(..) |
            ast::ItemConst(..) |
            ast::ItemFn(..) |
            ast::ItemMod(..) |
            ast::ItemForeignMod(..) |
            ast::ItemTy(..) |
            ast::ItemMac(..) => {
                visit::walk_item(self, item);
            }
        }
    }
}

// Constraint construction and representation
//
// The second pass over the AST determines the set of constraints.
// We walk the set of items and, for each member, generate new constraints.

struct ConstraintContext<'a, 'tcx: 'a> {
    terms_cx: TermsContext<'a, 'tcx>,

    // These are pointers to common `ConstantTerm` instances
    covariant: VarianceTermPtr<'a>,
    contravariant: VarianceTermPtr<'a>,
    invariant: VarianceTermPtr<'a>,
    bivariant: VarianceTermPtr<'a>,

    constraints: Vec<Constraint<'a>> ,
}

/// Declares that the variable `decl_id` appears in a location with
/// variance `variance`.
#[derive(Copy)]
struct Constraint<'a> {
    inferred: InferredIndex,
    variance: &'a VarianceTerm<'a>,
}

fn add_constraints_from_crate<'a, 'tcx>(terms_cx: TermsContext<'a, 'tcx>,
                                        krate: &ast::Crate)
                                        -> ConstraintContext<'a, 'tcx>
{
    let covariant = terms_cx.arena.alloc(ConstantTerm(ty::Covariant));
    let contravariant = terms_cx.arena.alloc(ConstantTerm(ty::Contravariant));
    let invariant = terms_cx.arena.alloc(ConstantTerm(ty::Invariant));
    let bivariant = terms_cx.arena.alloc(ConstantTerm(ty::Bivariant));
    let mut constraint_cx = ConstraintContext {
        terms_cx: terms_cx,
        covariant: covariant,
        contravariant: contravariant,
        invariant: invariant,
        bivariant: bivariant,
        constraints: Vec::new(),
    };
    visit::walk_crate(&mut constraint_cx, krate);
    constraint_cx
}

impl<'a, 'tcx, 'v> Visitor<'v> for ConstraintContext<'a, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        let did = ast_util::local_def(item.id);
        let tcx = self.terms_cx.tcx;

        debug!("visit_item item={}",
               item.repr(tcx));

        match item.node {
            ast::ItemEnum(ref enum_definition, _) => {
                let scheme = ty::lookup_item_type(tcx, did);

                // Not entirely obvious: constraints on structs/enums do not
                // affect the variance of their type parameters. See discussion
                // in comment at top of module.
                //
                // self.add_constraints_from_generics(&scheme.generics);

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
                for ast_variant in &enum_definition.variants {
                    let variant =
                        ty::VariantInfo::from_ast_variant(tcx,
                                                          &**ast_variant,
                                                          /*discriminant*/ 0);
                    for arg_ty in &variant.args {
                        self.add_constraints_from_ty(&scheme.generics, *arg_ty, self.covariant);
                    }
                }
            }

            ast::ItemStruct(..) => {
                let scheme = ty::lookup_item_type(tcx, did);

                // Not entirely obvious: constraints on structs/enums do not
                // affect the variance of their type parameters. See discussion
                // in comment at top of module.
                //
                // self.add_constraints_from_generics(&scheme.generics);

                let struct_fields = ty::lookup_struct_fields(tcx, did);
                for field_info in &struct_fields {
                    assert_eq!(field_info.id.krate, ast::LOCAL_CRATE);
                    let field_ty = ty::node_id_to_type(tcx, field_info.id.node);
                    self.add_constraints_from_ty(&scheme.generics, field_ty, self.covariant);
                }
            }

            ast::ItemTrait(..) => {
                let trait_def = ty::lookup_trait_def(tcx, did);
                let predicates = ty::predicates(tcx, ty::mk_self_type(tcx), &trait_def.bounds);
                self.add_constraints_from_predicates(&trait_def.generics,
                                                     &predicates[],
                                                     self.covariant);

                let trait_items = ty::trait_items(tcx, did);
                for trait_item in &*trait_items {
                    match *trait_item {
                        ty::MethodTraitItem(ref method) => {
                            self.add_constraints_from_predicates(
                                &method.generics,
                                method.predicates.predicates.get_slice(FnSpace),
                                self.contravariant);

                            self.add_constraints_from_sig(
                                &method.generics,
                                &method.fty.sig,
                                self.covariant);
                        }
                        ty::TypeTraitItem(_) => {}
                    }
                }
            }

            ast::ItemExternCrate(_) |
            ast::ItemUse(_) |
            ast::ItemStatic(..) |
            ast::ItemConst(..) |
            ast::ItemFn(..) |
            ast::ItemMod(..) |
            ast::ItemForeignMod(..) |
            ast::ItemTy(..) |
            ast::ItemImpl(..) |
            ast::ItemMac(..) => {
            }
        }

        visit::walk_item(self, item);
    }
}

/// Is `param_id` a lifetime according to `map`?
fn is_lifetime(map: &ast_map::Map, param_id: ast::NodeId) -> bool {
    match map.find(param_id) {
        Some(ast_map::NodeLifetime(..)) => true, _ => false
    }
}

impl<'a, 'tcx> ConstraintContext<'a, 'tcx> {
    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.terms_cx.tcx
    }

    fn inferred_index(&self, param_id: ast::NodeId) -> InferredIndex {
        match self.terms_cx.inferred_map.get(&param_id) {
            Some(&index) => index,
            None => {
                self.tcx().sess.bug(&format!(
                        "no inferred index entry for {}",
                        self.tcx().map.node_to_string(param_id))[]);
            }
        }
    }

    fn find_binding_for_lifetime(&self, param_id: ast::NodeId) -> ast::NodeId {
        let tcx = self.terms_cx.tcx;
        assert!(is_lifetime(&tcx.map, param_id));
        match tcx.named_region_map.get(&param_id) {
            Some(&rl::DefEarlyBoundRegion(_, _, lifetime_decl_id))
                => lifetime_decl_id,
            Some(_) => panic!("should not encounter non early-bound cases"),

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
                || panic!("tcx.map missing entry for id: {}", parent_id));

            let is_inferred;
            macro_rules! cannot_happen { () => { {
                panic!("invalid parent: {} for {}",
                      tcx.map.node_to_string(parent_id),
                      tcx.map.node_to_string(param_id));
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
                ast_map::NodeTraitItem(..)   => is_inferred = false,
                ast_map::NodeImplItem(..)    => is_inferred = false,
                _                            => cannot_happen!(),
            }

            return is_inferred;
        };

        assert_eq!(result, check_result(self));

        return result;
    }

    /// Returns a variance term representing the declared variance of the type/region parameter
    /// with the given id.
    fn declared_variance(&self,
                         param_def_id: ast::DefId,
                         item_def_id: ast::DefId,
                         kind: ParamKind,
                         space: ParamSpace,
                         index: uint)
                         -> VarianceTermPtr<'a> {
        assert_eq!(param_def_id.krate, item_def_id.krate);

        if param_def_id.krate == ast::LOCAL_CRATE {
            // Parameter on an item defined within current crate:
            // variance not yet inferred, so return a symbolic
            // variance.
            let InferredIndex(index) = self.inferred_index(param_def_id.node);
            self.terms_cx.inferred_infos[index].term
        } else {
            // Parameter on an item defined within another crate:
            // variance already inferred, just look it up.
            let variances = ty::item_variances(self.tcx(), item_def_id);
            let variance = match kind {
                TypeParam => *variances.types.get(space, index),
                RegionParam => *variances.regions.get(space, index),
            };
            self.constant_term(variance)
        }
    }

    fn add_constraint(&mut self,
                      InferredIndex(index): InferredIndex,
                      variance: VarianceTermPtr<'a>) {
        debug!("add_constraint(index={}, variance={:?})",
                index, variance);
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
                &*self.terms_cx.arena.alloc(TransformTerm(v1, v2))
            }
        }
    }

    fn add_constraints_from_trait_ref(&mut self,
                                      generics: &ty::Generics<'tcx>,
                                      trait_ref: &ty::TraitRef<'tcx>,
                                      variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_trait_ref: trait_ref={} variance={:?}",
               trait_ref.repr(self.tcx()),
               variance);

        let trait_def = ty::lookup_trait_def(self.tcx(), trait_ref.def_id);

        self.add_constraints_from_substs(
            generics,
            trait_ref.def_id,
            trait_def.generics.types.as_slice(),
            trait_def.generics.regions.as_slice(),
            trait_ref.substs,
            variance);
    }

    /// Adds constraints appropriate for an instance of `ty` appearing
    /// in a context with the generics defined in `generics` and
    /// ambient variance `variance`
    fn add_constraints_from_ty(&mut self,
                               generics: &ty::Generics<'tcx>,
                               ty: Ty<'tcx>,
                               variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_ty(ty={}, variance={:?})",
               ty.repr(self.tcx()),
               variance);

        match ty.sty {
            ty::ty_bool |
            ty::ty_char | ty::ty_int(_) | ty::ty_uint(_) |
            ty::ty_float(_) | ty::ty_str => {
                /* leaf type -- noop */
            }

            ty::ty_closure(..) => {
                self.tcx().sess.bug("Unexpected closure type in variance computation");
            }

            ty::ty_rptr(region, ref mt) => {
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(generics, *region, contra);
                self.add_constraints_from_mt(generics, mt, variance);
            }

            ty::ty_uniq(typ) | ty::ty_vec(typ, _) | ty::ty_open(typ) => {
                self.add_constraints_from_ty(generics, typ, variance);
            }


            ty::ty_ptr(ref mt) => {
                self.add_constraints_from_mt(generics, mt, variance);
            }

            ty::ty_tup(ref subtys) => {
                for &subty in subtys {
                    self.add_constraints_from_ty(generics, subty, variance);
                }
            }

            ty::ty_enum(def_id, substs) |
            ty::ty_struct(def_id, substs) => {
                let item_type = ty::lookup_item_type(self.tcx(), def_id);

                // All type parameters on enums and structs should be
                // in the TypeSpace.
                assert!(item_type.generics.types.is_empty_in(subst::SelfSpace));
                assert!(item_type.generics.types.is_empty_in(subst::FnSpace));
                assert!(item_type.generics.regions.is_empty_in(subst::SelfSpace));
                assert!(item_type.generics.regions.is_empty_in(subst::FnSpace));

                self.add_constraints_from_substs(
                    generics,
                    def_id,
                    item_type.generics.types.get_slice(subst::TypeSpace),
                    item_type.generics.regions.get_slice(subst::TypeSpace),
                    substs,
                    variance);
            }

            ty::ty_projection(ref data) => {
                let trait_ref = &data.trait_ref;
                let trait_def = ty::lookup_trait_def(self.tcx(), trait_ref.def_id);
                self.add_constraints_from_substs(
                    generics,
                    trait_ref.def_id,
                    trait_def.generics.types.as_slice(),
                    trait_def.generics.regions.as_slice(),
                    trait_ref.substs,
                    self.invariant);
            }

            ty::ty_trait(ref data) => {
                let poly_trait_ref =
                    data.principal_trait_ref_with_self_ty(self.tcx(),
                                                          self.tcx().types.err);

                // The type `Foo<T+'a>` is contravariant w/r/t `'a`:
                let contra = self.contravariant(variance);
                self.add_constraints_from_region(generics, data.bounds.region_bound, contra);

                // Ignore the SelfSpace, it is erased.
                self.add_constraints_from_trait_ref(generics, &*poly_trait_ref.0, variance);

                let projections = data.projection_bounds_with_self_ty(self.tcx(),
                                                                      self.tcx().types.err);
                for projection in &projections {
                    self.add_constraints_from_ty(generics, projection.0.ty, self.invariant);
                }
            }

            ty::ty_param(ref data) => {
                let def_id = generics.types.get(data.space, data.idx as uint).def_id;
                assert_eq!(def_id.krate, ast::LOCAL_CRATE);
                match self.terms_cx.inferred_map.get(&def_id.node) {
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

            ty::ty_bare_fn(_, &ty::BareFnTy { ref sig, .. }) => {
                self.add_constraints_from_sig(generics, sig, variance);
            }

            ty::ty_err => {
                // we encounter this when walking the trait references for object
                // types, where we use ty_err as the Self type
            }

            ty::ty_infer(..) => {
                self.tcx().sess.bug(
                    &format!("unexpected type encountered in \
                            variance inference: {}",
                            ty.repr(self.tcx()))[]);
            }
        }
    }


    /// Adds constraints appropriate for a nominal type (enum, struct,
    /// object, etc) appearing in a context with ambient variance `variance`
    fn add_constraints_from_substs(&mut self,
                                   generics: &ty::Generics<'tcx>,
                                   def_id: ast::DefId,
                                   type_param_defs: &[ty::TypeParameterDef<'tcx>],
                                   region_param_defs: &[ty::RegionParameterDef],
                                   substs: &subst::Substs<'tcx>,
                                   variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_substs(def_id={}, substs={}, variance={:?})",
               def_id.repr(self.tcx()),
               substs.repr(self.tcx()),
               variance);

        for p in type_param_defs {
            let variance_decl =
                self.declared_variance(p.def_id, def_id, TypeParam,
                                       p.space, p.index as uint);
            let variance_i = self.xform(variance, variance_decl);
            let substs_ty = *substs.types.get(p.space, p.index as uint);
            debug!("add_constraints_from_substs: variance_decl={:?} variance_i={:?}",
                   variance_decl, variance_i);
            self.add_constraints_from_ty(generics, substs_ty, variance_i);
        }

        for p in region_param_defs {
            let variance_decl =
                self.declared_variance(p.def_id, def_id,
                                       RegionParam, p.space, p.index as uint);
            let variance_i = self.xform(variance, variance_decl);
            let substs_r = *substs.regions().get(p.space, p.index as uint);
            self.add_constraints_from_region(generics, substs_r, variance_i);
        }
    }

    fn add_constraints_from_predicates(&mut self,
                                       generics: &ty::Generics<'tcx>,
                                       predicates: &[ty::Predicate<'tcx>],
                                       variance: VarianceTermPtr<'a>) {
        debug!("add_constraints_from_generics({})",
               generics.repr(self.tcx()));

        for predicate in predicates.iter() {
            match *predicate {
                ty::Predicate::Trait(ty::Binder(ref data)) => {
                    self.add_constraints_from_trait_ref(generics, &*data.trait_ref, variance);
                }

                ty::Predicate::Equate(ty::Binder(ref data)) => {
                    self.add_constraints_from_ty(generics, data.0, variance);
                    self.add_constraints_from_ty(generics, data.1, variance);
                }

                ty::Predicate::TypeOutlives(ty::Binder(ref data)) => {
                    self.add_constraints_from_ty(generics, data.0, variance);

                    let variance_r = self.xform(variance, self.contravariant);
                    self.add_constraints_from_region(generics, data.1, variance_r);
                }

                ty::Predicate::RegionOutlives(ty::Binder(ref data)) => {
                    // `'a : 'b` is still true if 'a gets bigger
                    self.add_constraints_from_region(generics, data.0, variance);

                    // `'a : 'b` is still true if 'b gets smaller
                    let variance_r = self.xform(variance, self.contravariant);
                    self.add_constraints_from_region(generics, data.1, variance_r);
                }

                ty::Predicate::Projection(ty::Binder(ref data)) => {
                    self.add_constraints_from_trait_ref(generics,
                                                        &*data.projection_ty.trait_ref,
                                                        variance);

                    self.add_constraints_from_ty(generics, data.ty, self.invariant);
                }
            }
        }
    }

    /// Adds constraints appropriate for a function with signature
    /// `sig` appearing in a context with ambient variance `variance`
    fn add_constraints_from_sig(&mut self,
                                generics: &ty::Generics<'tcx>,
                                sig: &ty::PolyFnSig<'tcx>,
                                variance: VarianceTermPtr<'a>) {
        let contra = self.contravariant(variance);
        for &input in &sig.0.inputs {
            self.add_constraints_from_ty(generics, input, contra);
        }
        if let ty::FnConverging(result_type) = sig.0.output {
            self.add_constraints_from_ty(generics, result_type, variance);
        }
    }

    /// Adds constraints appropriate for a region appearing in a
    /// context with ambient variance `variance`
    fn add_constraints_from_region(&mut self,
                                   _generics: &ty::Generics<'tcx>,
                                   region: ty::Region,
                                   variance: VarianceTermPtr<'a>) {
        match region {
            ty::ReEarlyBound(param_id, _, _, _) => {
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
                    .bug(&format!("unexpected region encountered in variance \
                                  inference: {}",
                                 region.repr(self.tcx()))[]);
            }
        }
    }

    /// Adds constraints appropriate for a mutability-type pair
    /// appearing in a context with ambient variance `variance`
    fn add_constraints_from_mt(&mut self,
                               generics: &ty::Generics<'tcx>,
                               mt: &ty::mt<'tcx>,
                               variance: VarianceTermPtr<'a>) {
        match mt.mutbl {
            ast::MutMutable => {
                let invar = self.invariant(variance);
                self.add_constraints_from_ty(generics, mt.ty, invar);
            }

            ast::MutImmutable => {
                self.add_constraints_from_ty(generics, mt.ty, variance);
            }
        }
    }
}

// Constraint solving
//
// The final phase iterates over the constraints, refining the variance
// for each inferred until a fixed point is reached. This will be the
// optimal solution to the constraints. The final variance for each
// inferred is then written into the `variance_map` in the tcx.

struct SolveContext<'a, 'tcx: 'a> {
    terms_cx: TermsContext<'a, 'tcx>,
    constraints: Vec<Constraint<'a>> ,

    // Maps from an InferredIndex to the inferred value for that variable.
    solutions: Vec<ty::Variance> }

fn solve_constraints(constraints_cx: ConstraintContext) {
    let ConstraintContext { terms_cx, constraints, .. } = constraints_cx;

    let solutions =
        terms_cx.inferred_infos.iter()
                               .map(|ii| ii.initial_variance)
                               .collect();

    let mut solutions_cx = SolveContext {
        terms_cx: terms_cx,
        constraints: constraints,
        solutions: solutions
    };
    solutions_cx.solve();
    solutions_cx.write();
}

impl<'a, 'tcx> SolveContext<'a, 'tcx> {
    fn solve(&mut self) {
        // Propagate constraints until a fixed point is reached.  Note
        // that the maximum number of iterations is 2C where C is the
        // number of constraints (each variable can change values at most
        // twice). Since number of constraints is linear in size of the
        // input, so is the inference process.
        let mut changed = true;
        while changed {
            changed = false;

            for constraint in &self.constraints {
                let Constraint { inferred, variance: term } = *constraint;
                let InferredIndex(inferred) = inferred;
                let variance = self.evaluate(term);
                let old_value = self.solutions[inferred];
                let new_value = glb(variance, old_value);
                if old_value != new_value {
                    debug!("Updating inferred {} (node {}) \
                            from {:?} to {:?} due to {:?}",
                            inferred,
                            self.terms_cx
                                .inferred_infos[inferred]
                                .param_id,
                            old_value,
                            new_value,
                            term);

                    self.solutions[inferred] = new_value;
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
            let item_id = inferred_infos[index].item_id;
            let mut types = VecPerParamSpace::empty();
            let mut regions = VecPerParamSpace::empty();

            while index < num_inferred && inferred_infos[index].item_id == item_id {
                let info = &inferred_infos[index];
                let variance = solutions[index];
                debug!("Index {} Info {} / {:?} / {:?} Variance {:?}",
                       index, info.index, info.kind, info.space, variance);
                match info.kind {
                    TypeParam => { types.push(info.space, variance); }
                    RegionParam => { regions.push(info.space, variance); }
                }

                index += 1;
            }

            let item_variances = ty::ItemVariances {
                types: types,
                regions: regions
            };
            debug!("item_id={} item_variances={}",
                    item_id,
                    item_variances.repr(tcx));

            let item_def_id = ast_util::local_def(item_id);

            // For unit testing: check for a special "rustc_variance"
            // attribute and report an error with various results if found.
            if ty::has_attr(tcx, item_def_id, "rustc_variance") {
                let found = item_variances.repr(tcx);
                span_err!(tcx.sess, tcx.map.span(item_id), E0208, "{}", &found[..]);
            }

            let newly_added = tcx.item_variance_map.borrow_mut()
                                 .insert(item_def_id, Rc::new(item_variances)).is_none();
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
                self.solutions[index]
            }
        }
    }
}

// Miscellany transformations on variance

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

