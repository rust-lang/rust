//! The type system. We currently use this to infer types for completion, hover
//! information and various assists.

mod autoderef;
pub(crate) mod primitive;
#[cfg(test)]
mod tests;
pub(crate) mod method_resolution;
mod op;
mod lower;
mod infer;
pub(crate) mod display;

use std::sync::Arc;
use std::{fmt, mem};

use crate::{Name, AdtDef, type_ref::Mutability, db::HirDatabase};

pub(crate) use lower::{TypableDef, CallableDef, type_for_def, type_for_field, callable_item_sig};
pub(crate) use infer::{infer, InferenceResult, InferTy};
use display::{HirDisplay, HirFormatter};

/// A type. This is based on the `TyKind` enum in rustc (librustc/ty/sty.rs).
///
/// This should be cheap to clone.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Ty {
    /// The primitive boolean type. Written as `bool`.
    Bool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point). Written as `char`.
    Char,

    /// A primitive integer type. For example, `i32`.
    Int(primitive::UncertainIntTy),

    /// A primitive floating-point type. For example, `f64`.
    Float(primitive::UncertainFloatTy),

    /// Structures, enumerations and unions.
    Adt {
        /// The definition of the struct/enum.
        def_id: AdtDef,
        /// Substitutions for the generic parameters of the type.
        substs: Substs,
    },

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// The pointee of an array slice.  Written as `[T]`.
    Slice(Arc<Ty>),

    /// An array with the given length. Written as `[T; n]`.
    Array(Arc<Ty>),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(Arc<Ty>, Mutability),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(Arc<Ty>, Mutability),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type, which is output (for a function
    /// named `foo` returning an `i32`) as `fn() -> i32 {foo}`.
    ///
    /// This includes tuple struct / enum variant constructors as well.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    FnDef {
        /// The definition of the function / constructor.
        def: CallableDef,
        /// Substitutions for the generic parameters of the type
        substs: Substs,
    },

    /// A pointer to a function.  Written as `fn() -> i32`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr(Substs),

    /// The never type `!`.
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple(Substs),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    Param {
        /// The index of the parameter (starting with parameters from the
        /// surrounding impl, then the current function).
        idx: u32,
        /// The name of the parameter, for displaying.
        name: Name,
    },

    /// A type variable used during type checking. Not to be confused with a
    /// type parameter.
    Infer(InferTy),

    /// A placeholder for a type which could not be computed; this is propagated
    /// to avoid useless error messages. Doubles as a placeholder where type
    /// variables are inserted before type checking, since we want to try to
    /// infer a better type here anyway -- for the IDE use case, we want to try
    /// to infer as much as possible even in the presence of type errors.
    Unknown,
}

/// A list of substitutions for generic parameters.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Substs(Arc<[Ty]>);

impl Substs {
    pub fn empty() -> Substs {
        Substs(Arc::new([]))
    }

    pub fn iter(&self) -> impl Iterator<Item = &Ty> {
        self.0.iter()
    }

    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        // Without an Arc::make_mut_slice, we can't avoid the clone here:
        let mut v: Vec<_> = self.0.iter().cloned().collect();
        for t in &mut v {
            t.walk_mut(f);
        }
        self.0 = v.into();
    }
}

/// A function signature.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnSig {
    params_and_return: Arc<[Ty]>,
}

impl FnSig {
    pub fn from_params_and_return(mut params: Vec<Ty>, ret: Ty) -> FnSig {
        params.push(ret);
        FnSig { params_and_return: params.into() }
    }

    pub fn from_fn_ptr_substs(substs: &Substs) -> FnSig {
        FnSig { params_and_return: Arc::clone(&substs.0) }
    }

    pub fn params(&self) -> &[Ty] {
        &self.params_and_return[0..self.params_and_return.len() - 1]
    }

    pub fn ret(&self) -> &Ty {
        &self.params_and_return[self.params_and_return.len() - 1]
    }

    pub fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        // Without an Arc::make_mut_slice, we can't avoid the clone here:
        let mut v: Vec<_> = self.params_and_return.iter().cloned().collect();
        for t in &mut v {
            t.walk_mut(f);
        }
        self.params_and_return = v.into();
    }
}

impl Ty {
    pub fn unit() -> Self {
        Ty::Tuple(Substs::empty())
    }

    pub fn walk(&self, f: &mut impl FnMut(&Ty)) {
        match self {
            Ty::Slice(t) | Ty::Array(t) => t.walk(f),
            Ty::RawPtr(t, _) => t.walk(f),
            Ty::Ref(t, _) => t.walk(f),
            Ty::Tuple(ts) => {
                for t in ts.iter() {
                    t.walk(f);
                }
            }
            Ty::FnPtr(sig) => {
                for t in sig.iter() {
                    t.walk(f);
                }
            }
            Ty::FnDef { substs, .. } => {
                for t in substs.0.iter() {
                    t.walk(f);
                }
            }
            Ty::Adt { substs, .. } => {
                for t in substs.0.iter() {
                    t.walk(f);
                }
            }
            Ty::Bool
            | Ty::Char
            | Ty::Int(_)
            | Ty::Float(_)
            | Ty::Str
            | Ty::Never
            | Ty::Param { .. }
            | Ty::Infer(_)
            | Ty::Unknown => {}
        }
        f(self);
    }

    fn walk_mut(&mut self, f: &mut impl FnMut(&mut Ty)) {
        match self {
            Ty::Slice(t) | Ty::Array(t) => Arc::make_mut(t).walk_mut(f),
            Ty::RawPtr(t, _) => Arc::make_mut(t).walk_mut(f),
            Ty::Ref(t, _) => Arc::make_mut(t).walk_mut(f),
            Ty::Tuple(ts) => {
                ts.walk_mut(f);
            }
            Ty::FnPtr(sig) => {
                sig.walk_mut(f);
            }
            Ty::FnDef { substs, .. } => {
                substs.walk_mut(f);
            }
            Ty::Adt { substs, .. } => {
                substs.walk_mut(f);
            }
            Ty::Bool
            | Ty::Char
            | Ty::Int(_)
            | Ty::Float(_)
            | Ty::Str
            | Ty::Never
            | Ty::Param { .. }
            | Ty::Infer(_)
            | Ty::Unknown => {}
        }
        f(self);
    }

    fn fold(mut self, f: &mut impl FnMut(Ty) -> Ty) -> Ty {
        self.walk_mut(&mut |ty_mut| {
            let ty = mem::replace(ty_mut, Ty::Unknown);
            *ty_mut = f(ty);
        });
        self
    }

    fn builtin_deref(&self) -> Option<Ty> {
        match self {
            Ty::Ref(t, _) => Some(Ty::clone(t)),
            Ty::RawPtr(t, _) => Some(Ty::clone(t)),
            _ => None,
        }
    }

    /// If this is a type with type parameters (an ADT or function), replaces
    /// the `Substs` for these type parameters with the given ones. (So e.g. if
    /// `self` is `Option<_>` and the substs contain `u32`, we'll have
    /// `Option<u32>` afterwards.)
    pub fn apply_substs(self, substs: Substs) -> Ty {
        match self {
            Ty::Adt { def_id, .. } => Ty::Adt { def_id, substs },
            Ty::FnDef { def, .. } => Ty::FnDef { def, substs },
            _ => self,
        }
    }

    /// Replaces type parameters in this type using the given `Substs`. (So e.g.
    /// if `self` is `&[T]`, where type parameter T has index 0, and the
    /// `Substs` contain `u32` at index 0, we'll have `&[u32]` afterwards.)
    pub fn subst(self, substs: &Substs) -> Ty {
        self.fold(&mut |ty| match ty {
            Ty::Param { idx, name } => {
                if (idx as usize) < substs.0.len() {
                    substs.0[idx as usize].clone()
                } else {
                    Ty::Param { idx, name }
                }
            }
            ty => ty,
        })
    }

    /// Returns the type parameters of this type if it has some (i.e. is an ADT
    /// or function); so if `self` is `Option<u32>`, this returns the `u32`.
    fn substs(&self) -> Option<Substs> {
        match self {
            Ty::Adt { substs, .. } | Ty::FnDef { substs, .. } => Some(substs.clone()),
            _ => None,
        }
    }
}

impl HirDisplay for &Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool")?,
            Ty::Char => write!(f, "char")?,
            Ty::Int(t) => write!(f, "{}", t)?,
            Ty::Float(t) => write!(f, "{}", t)?,
            Ty::Str => write!(f, "str")?,
            Ty::Slice(t) | Ty::Array(t) => {
                write!(f, "[{}]", t.display(f.db))?;
            }
            Ty::RawPtr(t, m) => {
                write!(f, "*{}{}", m.as_keyword_for_ptr(), t.display(f.db))?;
            }
            Ty::Ref(t, m) => {
                write!(f, "&{}{}", m.as_keyword_for_ref(), t.display(f.db))?;
            }
            Ty::Never => write!(f, "!")?,
            Ty::Tuple(ts) => {
                if ts.0.len() == 1 {
                    write!(f, "({},)", ts.0[0].display(f.db))?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(&*ts.0, ", ")?;
                    write!(f, ")")?;
                }
            }
            Ty::FnPtr(sig) => {
                let sig = FnSig::from_fn_ptr_substs(sig);
                write!(f, "fn(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            Ty::FnDef { def, substs, .. } => {
                let sig = f.db.callable_item_signature(*def);
                let name = match def {
                    CallableDef::Function(ff) => ff.name(f.db),
                    CallableDef::Struct(s) => s.name(f.db).unwrap_or_else(Name::missing),
                    CallableDef::EnumVariant(e) => e.name(f.db).unwrap_or_else(Name::missing),
                };
                match def {
                    CallableDef::Function(_) => write!(f, "fn {}", name)?,
                    CallableDef::Struct(_) | CallableDef::EnumVariant(_) => write!(f, "{}", name)?,
                }
                if substs.0.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*substs.0, ", ")?;
                    write!(f, ">")?;
                }
                write!(f, "(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            Ty::Adt { def_id, substs, .. } => {
                let name = match def_id {
                    AdtDef::Struct(s) => s.name(f.db),
                    AdtDef::Enum(e) => e.name(f.db),
                }
                .unwrap_or_else(Name::missing);
                write!(f, "{}", name)?;
                if substs.0.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*substs.0, ", ")?;
                    write!(f, ">")?;
                }
            }
            Ty::Param { name, .. } => write!(f, "{}", name)?,
            Ty::Unknown => write!(f, "{{unknown}}")?,
            Ty::Infer(..) => write!(f, "_")?,
        }
        Ok(())
    }
}
