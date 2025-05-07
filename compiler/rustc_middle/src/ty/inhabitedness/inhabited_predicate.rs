use rustc_macros::HashStable;
use smallvec::SmallVec;
use tracing::instrument;

use crate::ty::context::TyCtxt;
use crate::ty::{self, DefId, OpaqueTypeKey, Ty, TypingEnv};

/// Represents whether some type is inhabited in a given context.
/// Examples of uninhabited types are `!`, `enum Void {}`, or a struct
/// containing either of those types.
/// A type's inhabitedness may depend on the `ParamEnv` as well as what types
/// are visible in the current module.
#[derive(Clone, Copy, Debug, PartialEq, HashStable)]
pub enum InhabitedPredicate<'tcx> {
    /// Inhabited
    True,
    /// Uninhabited
    False,
    /// Uninhabited when a const value is non-zero. This occurs when there is an
    /// array of uninhabited items, but the array is inhabited if it is empty.
    ConstIsZero(ty::Const<'tcx>),
    /// Uninhabited if within a certain module. This occurs when an uninhabited
    /// type has restricted visibility.
    NotInModule(DefId),
    /// Inhabited if some generic type is inhabited.
    /// These are replaced by calling [`Self::instantiate`].
    GenericType(Ty<'tcx>),
    /// Inhabited if either we don't know the hidden type or we know it and it is inhabited.
    OpaqueType(OpaqueTypeKey<'tcx>),
    /// A AND B
    And(&'tcx [InhabitedPredicate<'tcx>; 2]),
    /// A OR B
    Or(&'tcx [InhabitedPredicate<'tcx>; 2]),
}

impl<'tcx> InhabitedPredicate<'tcx> {
    /// Returns true if the corresponding type is inhabited in the given `ParamEnv` and module.
    pub fn apply(
        self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        module_def_id: DefId,
    ) -> bool {
        self.apply_revealing_opaque(tcx, typing_env, module_def_id, &|_| None)
    }

    /// Returns true if the corresponding type is inhabited in the given `ParamEnv` and module,
    /// revealing opaques when possible.
    pub fn apply_revealing_opaque(
        self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        module_def_id: DefId,
        reveal_opaque: &impl Fn(OpaqueTypeKey<'tcx>) -> Option<Ty<'tcx>>,
    ) -> bool {
        let Ok(result) = self.apply_inner::<!>(
            tcx,
            typing_env,
            &mut Default::default(),
            &|id| Ok(tcx.is_descendant_of(module_def_id, id)),
            reveal_opaque,
        );
        result
    }

    /// Same as `apply`, but returns `None` if self contains a module predicate
    pub fn apply_any_module(self, tcx: TyCtxt<'tcx>, typing_env: TypingEnv<'tcx>) -> Option<bool> {
        self.apply_inner(tcx, typing_env, &mut Default::default(), &|_| Err(()), &|_| None).ok()
    }

    /// Same as `apply`, but `NotInModule(_)` predicates yield `false`. That is,
    /// privately uninhabited types are considered always uninhabited.
    pub fn apply_ignore_module(self, tcx: TyCtxt<'tcx>, typing_env: TypingEnv<'tcx>) -> bool {
        let Ok(result) =
            self.apply_inner::<!>(tcx, typing_env, &mut Default::default(), &|_| Ok(true), &|_| {
                None
            });
        result
    }

    #[instrument(level = "debug", skip(tcx, typing_env, in_module, reveal_opaque), ret)]
    fn apply_inner<E: std::fmt::Debug>(
        self,
        tcx: TyCtxt<'tcx>,
        typing_env: TypingEnv<'tcx>,
        eval_stack: &mut SmallVec<[Ty<'tcx>; 1]>, // for cycle detection
        in_module: &impl Fn(DefId) -> Result<bool, E>,
        reveal_opaque: &impl Fn(OpaqueTypeKey<'tcx>) -> Option<Ty<'tcx>>,
    ) -> Result<bool, E> {
        match self {
            Self::False => Ok(false),
            Self::True => Ok(true),
            Self::ConstIsZero(const_) => match const_.try_to_target_usize(tcx) {
                None | Some(0) => Ok(true),
                Some(1..) => Ok(false),
            },
            Self::NotInModule(id) => in_module(id).map(|in_mod| !in_mod),
            // `t` may be a projection, for which `inhabited_predicate` returns a `GenericType`. As
            // we have a param_env available, we can do better.
            Self::GenericType(t) => {
                let normalized_pred = tcx
                    .try_normalize_erasing_regions(typing_env, t)
                    .map_or(self, |t| t.inhabited_predicate(tcx));
                match normalized_pred {
                    // We don't have more information than we started with, so consider inhabited.
                    Self::GenericType(_) => Ok(true),
                    pred => {
                        // A type which is cyclic when monomorphized can happen here since the
                        // layout error would only trigger later. See e.g. `tests/ui/sized/recursive-type-2.rs`.
                        if eval_stack.contains(&t) {
                            return Ok(true); // Recover; this will error later.
                        }
                        eval_stack.push(t);
                        let ret =
                            pred.apply_inner(tcx, typing_env, eval_stack, in_module, reveal_opaque);
                        eval_stack.pop();
                        ret
                    }
                }
            }
            Self::OpaqueType(key) => match reveal_opaque(key) {
                // Unknown opaque is assumed inhabited.
                None => Ok(true),
                // Known opaque type is inspected recursively.
                Some(t) => {
                    // A cyclic opaque type can happen in corner cases that would only error later.
                    // See e.g. `tests/ui/type-alias-impl-trait/recursive-tait-conflicting-defn.rs`.
                    if eval_stack.contains(&t) {
                        return Ok(true); // Recover; this will error later.
                    }
                    eval_stack.push(t);
                    let ret = t.inhabited_predicate(tcx).apply_inner(
                        tcx,
                        typing_env,
                        eval_stack,
                        in_module,
                        reveal_opaque,
                    );
                    eval_stack.pop();
                    ret
                }
            },
            Self::And([a, b]) => try_and(a, b, |x| {
                x.apply_inner(tcx, typing_env, eval_stack, in_module, reveal_opaque)
            }),
            Self::Or([a, b]) => try_or(a, b, |x| {
                x.apply_inner(tcx, typing_env, eval_stack, in_module, reveal_opaque)
            }),
        }
    }

    pub fn and(self, tcx: TyCtxt<'tcx>, other: Self) -> Self {
        self.reduce_and(tcx, other).unwrap_or_else(|| Self::And(tcx.arena.alloc([self, other])))
    }

    pub fn or(self, tcx: TyCtxt<'tcx>, other: Self) -> Self {
        self.reduce_or(tcx, other).unwrap_or_else(|| Self::Or(tcx.arena.alloc([self, other])))
    }

    pub fn all(tcx: TyCtxt<'tcx>, iter: impl IntoIterator<Item = Self>) -> Self {
        let mut result = Self::True;
        for pred in iter {
            if matches!(pred, Self::False) {
                return Self::False;
            }
            result = result.and(tcx, pred);
        }
        result
    }

    pub fn any(tcx: TyCtxt<'tcx>, iter: impl IntoIterator<Item = Self>) -> Self {
        let mut result = Self::False;
        for pred in iter {
            if matches!(pred, Self::True) {
                return Self::True;
            }
            result = result.or(tcx, pred);
        }
        result
    }

    fn reduce_and(self, tcx: TyCtxt<'tcx>, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::True, a) | (a, Self::True) => Some(a),
            (Self::False, _) | (_, Self::False) => Some(Self::False),
            (Self::ConstIsZero(a), Self::ConstIsZero(b)) if a == b => Some(Self::ConstIsZero(a)),
            (Self::NotInModule(a), Self::NotInModule(b)) if a == b => Some(Self::NotInModule(a)),
            (Self::NotInModule(a), Self::NotInModule(b)) if tcx.is_descendant_of(a, b) => {
                Some(Self::NotInModule(b))
            }
            (Self::NotInModule(a), Self::NotInModule(b)) if tcx.is_descendant_of(b, a) => {
                Some(Self::NotInModule(a))
            }
            (Self::GenericType(a), Self::GenericType(b)) if a == b => Some(Self::GenericType(a)),
            (Self::And(&[a, b]), c) | (c, Self::And(&[a, b])) => {
                if let Some(ac) = a.reduce_and(tcx, c) {
                    Some(ac.and(tcx, b))
                } else if let Some(bc) = b.reduce_and(tcx, c) {
                    Some(Self::And(tcx.arena.alloc([a, bc])))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn reduce_or(self, tcx: TyCtxt<'tcx>, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::True, _) | (_, Self::True) => Some(Self::True),
            (Self::False, a) | (a, Self::False) => Some(a),
            (Self::ConstIsZero(a), Self::ConstIsZero(b)) if a == b => Some(Self::ConstIsZero(a)),
            (Self::NotInModule(a), Self::NotInModule(b)) if a == b => Some(Self::NotInModule(a)),
            (Self::NotInModule(a), Self::NotInModule(b)) if tcx.is_descendant_of(a, b) => {
                Some(Self::NotInModule(a))
            }
            (Self::NotInModule(a), Self::NotInModule(b)) if tcx.is_descendant_of(b, a) => {
                Some(Self::NotInModule(b))
            }
            (Self::GenericType(a), Self::GenericType(b)) if a == b => Some(Self::GenericType(a)),
            (Self::Or(&[a, b]), c) | (c, Self::Or(&[a, b])) => {
                if let Some(ac) = a.reduce_or(tcx, c) {
                    Some(ac.or(tcx, b))
                } else if let Some(bc) = b.reduce_or(tcx, c) {
                    Some(Self::Or(tcx.arena.alloc([a, bc])))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Replaces generic types with its corresponding predicate
    pub fn instantiate(self, tcx: TyCtxt<'tcx>, args: ty::GenericArgsRef<'tcx>) -> Self {
        self.instantiate_opt(tcx, args).unwrap_or(self)
    }

    /// Same as [`Self::instantiate`], but if there is no generics to
    /// instantiate, returns `None`. This is useful because it lets us avoid
    /// allocating a recursive copy of everything when the result is unchanged.
    ///
    /// Only used to implement `instantiate` itself.
    fn instantiate_opt(self, tcx: TyCtxt<'tcx>, args: ty::GenericArgsRef<'tcx>) -> Option<Self> {
        match self {
            Self::ConstIsZero(c) => {
                let c = ty::EarlyBinder::bind(c).instantiate(tcx, args);
                let pred = match c.try_to_target_usize(tcx) {
                    Some(0) => Self::True,
                    Some(1..) => Self::False,
                    None => Self::ConstIsZero(c),
                };
                Some(pred)
            }
            Self::GenericType(t) => {
                Some(ty::EarlyBinder::bind(t).instantiate(tcx, args).inhabited_predicate(tcx))
            }
            Self::And(&[a, b]) => match a.instantiate_opt(tcx, args) {
                None => b.instantiate_opt(tcx, args).map(|b| a.and(tcx, b)),
                Some(InhabitedPredicate::False) => Some(InhabitedPredicate::False),
                Some(a) => Some(a.and(tcx, b.instantiate_opt(tcx, args).unwrap_or(b))),
            },
            Self::Or(&[a, b]) => match a.instantiate_opt(tcx, args) {
                None => b.instantiate_opt(tcx, args).map(|b| a.or(tcx, b)),
                Some(InhabitedPredicate::True) => Some(InhabitedPredicate::True),
                Some(a) => Some(a.or(tcx, b.instantiate_opt(tcx, args).unwrap_or(b))),
            },
            Self::True | Self::False | Self::NotInModule(_) => None,
            Self::OpaqueType(_) => {
                bug!("unexpected OpaqueType in InhabitedPredicate");
            }
        }
    }
}

// this is basically like `f(a)? && f(b)?` but different in the case of
// `Ok(false) && Err(_) -> Ok(false)`
fn try_and<T, E>(a: T, b: T, mut f: impl FnMut(T) -> Result<bool, E>) -> Result<bool, E> {
    let a = f(a);
    if matches!(a, Ok(false)) {
        return Ok(false);
    }
    match (a, f(b)) {
        (_, Ok(false)) | (Ok(false), _) => Ok(false),
        (Ok(true), Ok(true)) => Ok(true),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

fn try_or<T, E>(a: T, b: T, mut f: impl FnMut(T) -> Result<bool, E>) -> Result<bool, E> {
    let a = f(a);
    if matches!(a, Ok(true)) {
        return Ok(true);
    }
    match (a, f(b)) {
        (_, Ok(true)) | (Ok(true), _) => Ok(true),
        (Ok(false), Ok(false)) => Ok(false),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}
