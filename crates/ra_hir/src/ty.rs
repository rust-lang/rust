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

use std::sync::Arc;
use std::{fmt, mem};

use join_to_string::join;

use crate::{Name, AdtDef, type_ref::Mutability};

pub(crate) use lower::{TypableDef, CallableDef, type_for_def, type_for_field};
pub(crate) use infer::{infer, InferenceResult, InferTy};

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
        /// The name, for displaying.
        name: Name,
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
        /// For display
        name: Name,
        /// Parameters and return type
        sig: Arc<FnSig>,
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
    FnPtr(Arc<FnSig>),

    /// The never type `!`.
    Never,

    /// A tuple type.  For example, `(i32, bool)`.
    Tuple(Arc<[Ty]>),

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
}

/// A function signature.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnSig {
    input: Vec<Ty>,
    output: Ty,
}

impl Ty {
    pub fn unit() -> Self {
        Ty::Tuple(Arc::new([]))
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
                for input in &sig.input {
                    input.walk(f);
                }
                sig.output.walk(f);
            }
            Ty::FnDef { substs, sig, .. } => {
                for input in &sig.input {
                    input.walk(f);
                }
                sig.output.walk(f);
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
                // Without an Arc::make_mut_slice, we can't avoid the clone here:
                let mut v: Vec<_> = ts.iter().cloned().collect();
                for t in &mut v {
                    t.walk_mut(f);
                }
                *ts = v.into();
            }
            Ty::FnPtr(sig) => {
                let sig_mut = Arc::make_mut(sig);
                for input in &mut sig_mut.input {
                    input.walk_mut(f);
                }
                sig_mut.output.walk_mut(f);
            }
            Ty::FnDef { substs, sig, .. } => {
                let sig_mut = Arc::make_mut(sig);
                for input in &mut sig_mut.input {
                    input.walk_mut(f);
                }
                sig_mut.output.walk_mut(f);
                // Without an Arc::make_mut_slice, we can't avoid the clone here:
                let mut v: Vec<_> = substs.0.iter().cloned().collect();
                for t in &mut v {
                    t.walk_mut(f);
                }
                substs.0 = v.into();
            }
            Ty::Adt { substs, .. } => {
                // Without an Arc::make_mut_slice, we can't avoid the clone here:
                let mut v: Vec<_> = substs.0.iter().cloned().collect();
                for t in &mut v {
                    t.walk_mut(f);
                }
                substs.0 = v.into();
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
            Ty::Adt { def_id, name, .. } => Ty::Adt { def_id, name, substs },
            Ty::FnDef { def, name, sig, .. } => Ty::FnDef { def, name, sig, substs },
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

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Ty::Bool => write!(f, "bool"),
            Ty::Char => write!(f, "char"),
            Ty::Int(t) => write!(f, "{}", t.ty_to_string()),
            Ty::Float(t) => write!(f, "{}", t.ty_to_string()),
            Ty::Str => write!(f, "str"),
            Ty::Slice(t) | Ty::Array(t) => write!(f, "[{}]", t),
            Ty::RawPtr(t, m) => write!(f, "*{}{}", m.as_keyword_for_ptr(), t),
            Ty::Ref(t, m) => write!(f, "&{}{}", m.as_keyword_for_ref(), t),
            Ty::Never => write!(f, "!"),
            Ty::Tuple(ts) => {
                if ts.len() == 1 {
                    write!(f, "({},)", ts[0])
                } else {
                    join(ts.iter()).surround_with("(", ")").separator(", ").to_fmt(f)
                }
            }
            Ty::FnPtr(sig) => {
                join(sig.input.iter()).surround_with("fn(", ")").separator(", ").to_fmt(f)?;
                write!(f, " -> {}", sig.output)
            }
            Ty::FnDef { def, name, substs, sig, .. } => {
                match def {
                    CallableDef::Function(_) => write!(f, "fn {}", name)?,
                    CallableDef::Struct(_) | CallableDef::EnumVariant(_) => write!(f, "{}", name)?,
                }
                if substs.0.len() > 0 {
                    join(substs.0.iter()).surround_with("<", ">").separator(", ").to_fmt(f)?;
                }
                join(sig.input.iter()).surround_with("(", ")").separator(", ").to_fmt(f)?;
                write!(f, " -> {}", sig.output)
            }
            Ty::Adt { name, substs, .. } => {
                write!(f, "{}", name)?;
                if substs.0.len() > 0 {
                    join(substs.0.iter()).surround_with("<", ">").separator(", ").to_fmt(f)?;
                }
                Ok(())
            }
            Ty::Param { name, .. } => write!(f, "{}", name),
            Ty::Unknown => write!(f, "[unknown]"),
            Ty::Infer(..) => write!(f, "_"),
        }
    }
}
