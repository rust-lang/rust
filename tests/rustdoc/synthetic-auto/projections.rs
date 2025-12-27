// FIXME(fmease): Early mock-up. Expectations are temporary.
#![crate_name = "it"]

pub struct Outer<T>(Inner<T>);
struct Inner<T>(T);

//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T> Unpin for Outer<T>where T: Iterator<Item = ()>"

impl<T> Unpin for Inner<T>
where
    T: Iterator<Item = ()>
{}

// FIXME: Nicer name for the synthetic param.
//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T, X0> UnwindSafe for Outer<T>where T: Iterator<Item = X0>"

impl<T, U> std::panic::UnwindSafe for Inner<T>
where
    T: Iterator<Item = U>
{}

// FIXME: Nicer name for the synthetic param.
//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T, X0> Send for Outer<T>\
//       where \
//           T: Iterator<Item = X0>, \
//           X0: Copy"

unsafe impl<T, U> Send for Inner<T>
where
    T: Iterator<Item = U>,
    U: Copy,
{}

// FIXME: Nicer name for the synthetic param.
// FIXME: Ugly + superfluous `?Sized` bound.
//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T, X0> Sync for Outer<T>\
//       where \
//           T: Iterator<Item = X0>, \
//           X0: Copy + ?Sized"

unsafe impl<T> Sync for Inner<T>
where
    T: Iterator<Item: Copy>
{}

// FIXME: Utter downfall, bound vomit
//@ has it/struct.Outer.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T, X0, X1, X2> RefUnwindSafe for Outer<T>\
//       where \
//           T: Iterator<Item = X0, Item = X2>, \
//           X2: Iterator + ?Sized, \
//           X0: ?Sized, \
//           <X0 as Iterator>::Item == X1"

impl<T, U> std::panic::RefUnwindSafe for Inner<T>
where
    T: Iterator<Item: Iterator<Item = U>>
{}

/* TREE:
(G) Err(NoSolution)  Misc  for[] TraitPredicate(<Outer<T> as std::panic::RefUnwindSafe>, polarity:Positive)
  (C) Err(NoSolution)  TraitCandidate/BuiltinImpl(Misc)
    (G) Err(NoSolution)  ImplWhereBound  for[] TraitPredicate(<Inner<T> as std::panic::RefUnwindSafe>, polarity:Positive)
      (C) Err(NoSolution)  TraitCandidate/Impl(DefId(0:21 ~ it[c9cb]::{impl#4}))
        (G) Err(NoSolution)  NormalizeGoal(Coinductive)  for[] AliasRelate(Term::Ty(Alias(Projection, AliasTy { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. })), Equate, Term::Ty(?2t))
          (C) Err(NoSolution)  Root { result: Err(NoSolution) }
            (G) Err(NoSolution)  TypeRelating  for[] NormalizesTo(AliasTerm { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?31t))
        (G) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  ImplWhereBound  for[] ProjectionPredicate(AliasTerm { args: [?2t], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?1t))
          (C) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  Root { result: Ok(Canonical { value: Response { certainty: Maybe { cause: Ambiguity, opaque_types_jank: AllGood }, var_values: CanonicalVarValues { var_values: [^c_0, ^c_1, ^c_2] }, external_constraints: ExternalConstraints(ExternalConstraintsData { region_constraints: [], opaque_types: [], normalization_nested_goals: NestedNormalizationGoals([]) }) }, max_universe: U0, variables: [PlaceholderTy(!0), Ty { ui: U0, sub_root: 1 }, Ty { ui: U0, sub_root: 2 }] }) }
            (G) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  TypeRelating  for[] AliasRelate(Term::Ty(Alias(Projection, AliasTy { args: [?2t], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. })), Equate, Term::Ty(?1t))
              (C) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  Root { result: Ok(Canonical { value: Response { certainty: Maybe { cause: Ambiguity, opaque_types_jank: AllGood }, var_values: CanonicalVarValues { var_values: [^c_0, ^c_1, ^c_2] }, external_constraints: ExternalConstraints(ExternalConstraintsData { region_constraints: [], opaque_types: [], normalization_nested_goals: NestedNormalizationGoals([]) }) }, max_universe: U0, variables: [PlaceholderTy(!0), Ty { ui: U0, sub_root: 1 }, Ty { ui: U0, sub_root: 2 }] }) }
                (G) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  TypeRelating  for[] NormalizesTo(AliasTerm { args: [?2t], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?36t))
                  (C) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  TraitCandidate/BuiltinImpl(Misc)
        (G) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  ImplWhereBound  for[] TraitPredicate(<_ as std::marker::Sized>, polarity:Positive)
          (C) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  TraitCandidate/BuiltinImpl(Misc)
        (G) Err(NoSolution)  ImplWhereBound  for[] TraitPredicate(<T as std::iter::Iterator>, polarity:Positive)
        (G) Err(NoSolution)  NormalizeGoal(Coinductive)  for[] AliasRelate(Term::Ty(Alias(Projection, AliasTy { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. })), Equate, Term::Ty(?11t))
          (C) Err(NoSolution)  Root { result: Err(NoSolution) }
            (G) Err(NoSolution)  TypeRelating  for[] NormalizesTo(AliasTerm { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?31t))
        (G) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  ImplWhereBound  for[] TraitPredicate(<_ as std::iter::Iterator>, polarity:Positive)
          (C) Ok(Maybe { cause: Ambiguity, opaque_types_jank: AllGood })  TraitCandidate/BuiltinImpl(Misc)
        (G) Ok(Yes)  ImplWhereBound  for[] TraitPredicate(<T as std::marker::Sized>, polarity:Positive)
          (C) Ok(Yes)  TraitCandidate/ParamEnv(NonGlobal)
*/
/* RESULT:
Binder { value: ProjectionPredicate(AliasTerm { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?2t)), bound_vars: [] },
Binder { value: ProjectionPredicate(AliasTerm { args: [?2t], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?1t)), bound_vars: [] },
Binder { value: TraitPredicate(<_ as std::marker::Sized>, polarity:Positive), bound_vars: [] },
Binder { value: TraitPredicate(<T as std::iter::Iterator>, polarity:Positive), bound_vars: [] },
Binder { value: ProjectionPredicate(AliasTerm { args: [T/#0], def_id: DefId(2:9696 ~ core[efc6]::iter::traits::iterator::Iterator::Item), .. }, Term::Ty(?11t)), bound_vars: [] },
Binder { value: TraitPredicate(<_ as std::iter::Iterator>, polarity:Positive), bound_vars: [] },
Binder { value: TraitPredicate(<T as std::marker::Sized>, polarity:Positive), bound_vars: [] },
*/

//////////////////////////////////////////////////////////////

pub struct Extern<T>(Intern<T>);
struct Intern<T>(T);

// FIXME: Well, some improvmenets can&should be done in clean/mod, resugaring more eq preds.

//@ has it/struct.Extern.html \
//      '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//      "impl<T, X0> Unpin for Extern<T>\
//       where \
//           T: Trait, \
//           X0: Copy + ?Sized, \
//           <<T as Trait>::Assoc as Trait>::Assoc == X0,"

impl<T> Unpin for Intern<T>
where
    T: Trait,
    <T::Assoc as Trait>::Assoc: Copy,
{}

trait Trait { type Assoc: Trait; }
