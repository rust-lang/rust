//! A desugared representation of paths like `crate::foo` or `<Type as Trait>::bar`.

use std::iter;

use crate::{
    lang_item::LangItemTarget,
    type_ref::{ConstRef, LifetimeRefId, TypeBound, TypeRefId},
};
use hir_expand::{
    mod_path::{ModPath, PathKind},
    name::Name,
};
use intern::Interned;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Path {
    /// `BarePath` is used when the path has neither generics nor type anchor, since the vast majority of paths
    /// are in this category, and splitting `Path` this way allows it to be more thin. When the path has either generics
    /// or type anchor, it is `Path::Normal` with the generics filled with `None` even if there are none (practically
    /// this is not a problem since many more paths have generics than a type anchor).
    BarePath(Interned<ModPath>),
    /// `Path::Normal` will always have either generics or type anchor.
    Normal(Box<NormalPath>),
    /// A link to a lang item. It is used in desugaring of things like `it?`. We can show these
    /// links via a normal path since they might be private and not accessible in the usage place.
    LangItem(LangItemTarget, Option<Name>),
}

// This type is being used a lot, make sure it doesn't grow unintentionally.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
const _: () = {
    assert!(size_of::<Path>() == 24);
    assert!(size_of::<Option<Path>>() == 24);
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalPath {
    pub generic_args: Box<[Option<GenericArgs>]>,
    pub type_anchor: Option<TypeRefId>,
    pub mod_path: Interned<ModPath>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenericArgsParentheses {
    No,
    /// Bounds of the form `Type::method(..): Send` or `impl Trait<method(..): Send>`,
    /// aka. Return Type Notation or RTN.
    ReturnTypeNotation,
    /// `Fn`-family parenthesized traits, e.g. `impl Fn(u32) -> String`.
    ///
    /// This is desugared into one generic argument containing a tuple of all arguments,
    /// and an associated type binding for `Output` for the return type.
    ParenSugar,
}

/// Generic arguments to a path segment (e.g. the `i32` in `Option<i32>`). This
/// also includes bindings of associated types, like in `Iterator<Item = Foo>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericArgs {
    pub args: Box<[GenericArg]>,
    /// This specifies whether the args contain a Self type as the first
    /// element. This is the case for path segments like `<T as Trait>`, where
    /// `T` is actually a type parameter for the path `Trait` specifying the
    /// Self type. Otherwise, when we have a path `Trait<X, Y>`, the Self type
    /// is left out.
    pub has_self_type: bool,
    /// Associated type bindings like in `Iterator<Item = T>`.
    pub bindings: Box<[AssociatedTypeBinding]>,
    /// Whether these generic args were written with parentheses and how.
    pub parenthesized: GenericArgsParentheses,
}

/// An associated type binding like in `Iterator<Item = T>`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AssociatedTypeBinding {
    /// The name of the associated type.
    pub name: Name,
    /// The generic arguments to the associated type. e.g. For `Trait<Assoc<'a, T> = &'a T>`, this
    /// would be `['a, T]`.
    pub args: Option<GenericArgs>,
    /// The type bound to this associated type (in `Item = T`, this would be the
    /// `T`). This can be `None` if there are bounds instead.
    pub type_ref: Option<TypeRefId>,
    /// Bounds for the associated type, like in `Iterator<Item:
    /// SomeOtherTrait>`. (This is the unstable `associated_type_bounds`
    /// feature.)
    pub bounds: Box<[TypeBound]>,
}

/// A single generic argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericArg {
    Type(TypeRefId),
    Lifetime(LifetimeRefId),
    Const(ConstRef),
}

impl Path {
    /// Converts a known mod path to `Path`.
    pub fn from_known_path(path: ModPath, generic_args: Vec<Option<GenericArgs>>) -> Path {
        Path::Normal(Box::new(NormalPath {
            generic_args: generic_args.into_boxed_slice(),
            type_anchor: None,
            mod_path: Interned::new(path),
        }))
    }

    /// Converts a known mod path to `Path`.
    pub fn from_known_path_with_no_generic(path: ModPath) -> Path {
        Path::BarePath(Interned::new(path))
    }

    #[inline]
    pub fn kind(&self) -> &PathKind {
        match self {
            Path::BarePath(mod_path) => &mod_path.kind,
            Path::Normal(path) => &path.mod_path.kind,
            Path::LangItem(..) => &PathKind::Abs,
        }
    }

    #[inline]
    pub fn type_anchor(&self) -> Option<TypeRefId> {
        match self {
            Path::Normal(path) => path.type_anchor,
            Path::LangItem(..) | Path::BarePath(_) => None,
        }
    }

    #[inline]
    pub fn generic_args(&self) -> Option<&[Option<GenericArgs>]> {
        match self {
            Path::Normal(path) => Some(&path.generic_args),
            Path::LangItem(..) | Path::BarePath(_) => None,
        }
    }

    pub fn segments(&self) -> PathSegments<'_> {
        match self {
            Path::BarePath(mod_path) => {
                PathSegments { segments: mod_path.segments(), generic_args: None }
            }
            Path::Normal(path) => PathSegments {
                segments: path.mod_path.segments(),
                generic_args: Some(&path.generic_args),
            },
            Path::LangItem(_, seg) => PathSegments { segments: seg.as_slice(), generic_args: None },
        }
    }

    pub fn mod_path(&self) -> Option<&ModPath> {
        match self {
            Path::BarePath(mod_path) => Some(mod_path),
            Path::Normal(path) => Some(&path.mod_path),
            Path::LangItem(..) => None,
        }
    }

    pub fn qualifier(&self) -> Option<Path> {
        match self {
            Path::BarePath(mod_path) => {
                if mod_path.is_ident() {
                    return None;
                }
                Some(Path::BarePath(Interned::new(ModPath::from_segments(
                    mod_path.kind,
                    mod_path.segments()[..mod_path.segments().len() - 1].iter().cloned(),
                ))))
            }
            Path::Normal(path) => {
                let mod_path = &path.mod_path;
                if mod_path.is_ident() {
                    return None;
                }
                let type_anchor = path.type_anchor;
                let generic_args = &path.generic_args;
                let qualifier_mod_path = Interned::new(ModPath::from_segments(
                    mod_path.kind,
                    mod_path.segments()[..mod_path.segments().len() - 1].iter().cloned(),
                ));
                let qualifier_generic_args = &generic_args[..generic_args.len() - 1];
                if type_anchor.is_none() && qualifier_generic_args.iter().all(|it| it.is_none()) {
                    Some(Path::BarePath(qualifier_mod_path))
                } else {
                    Some(Path::Normal(Box::new(NormalPath {
                        type_anchor,
                        mod_path: qualifier_mod_path,
                        generic_args: qualifier_generic_args.iter().cloned().collect(),
                    })))
                }
            }
            Path::LangItem(..) => None,
        }
    }

    pub fn is_self_type(&self) -> bool {
        match self {
            Path::BarePath(mod_path) => mod_path.is_Self(),
            Path::Normal(path) => {
                path.type_anchor.is_none()
                    && path.mod_path.is_Self()
                    && path.generic_args.iter().all(|args| args.is_none())
            }
            Path::LangItem(..) => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathSegment<'a> {
    pub name: &'a Name,
    pub args_and_bindings: Option<&'a GenericArgs>,
}

impl PathSegment<'_> {
    pub const MISSING: PathSegment<'static> =
        PathSegment { name: &Name::missing(), args_and_bindings: None };
}

#[derive(Debug, Clone, Copy)]
pub struct PathSegments<'a> {
    segments: &'a [Name],
    generic_args: Option<&'a [Option<GenericArgs>]>,
}

impl<'a> PathSegments<'a> {
    pub const EMPTY: PathSegments<'static> = PathSegments { segments: &[], generic_args: None };
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn len(&self) -> usize {
        self.segments.len()
    }
    pub fn first(&self) -> Option<PathSegment<'a>> {
        self.get(0)
    }
    pub fn last(&self) -> Option<PathSegment<'a>> {
        self.get(self.len().checked_sub(1)?)
    }

    pub fn get(&self, idx: usize) -> Option<PathSegment<'a>> {
        let res = PathSegment {
            name: self.segments.get(idx)?,
            args_and_bindings: self.generic_args.and_then(|it| it.get(idx)?.as_ref()),
        };
        Some(res)
    }

    pub fn skip(&self, len: usize) -> PathSegments<'a> {
        PathSegments {
            segments: self.segments.get(len..).unwrap_or(&[]),
            generic_args: self.generic_args.and_then(|it| it.get(len..)),
        }
    }

    pub fn take(&self, len: usize) -> PathSegments<'a> {
        PathSegments {
            segments: self.segments.get(..len).unwrap_or(self.segments),
            generic_args: self.generic_args.map(|it| it.get(..len).unwrap_or(it)),
        }
    }

    pub fn strip_last(&self) -> PathSegments<'a> {
        PathSegments {
            segments: self.segments.split_last().map_or(&[], |it| it.1),
            generic_args: self.generic_args.map(|it| it.split_last().map_or(&[][..], |it| it.1)),
        }
    }

    pub fn strip_last_two(&self) -> PathSegments<'a> {
        PathSegments {
            segments: self.segments.get(..self.segments.len().saturating_sub(2)).unwrap_or(&[]),
            generic_args: self
                .generic_args
                .map(|it| it.get(..it.len().saturating_sub(2)).unwrap_or(&[])),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = PathSegment<'a>> {
        self.segments
            .iter()
            .zip(self.generic_args.into_iter().flatten().chain(iter::repeat(&None)))
            .map(|(name, args)| PathSegment { name, args_and_bindings: args.as_ref() })
    }
}

impl GenericArgs {
    pub(crate) fn empty() -> GenericArgs {
        GenericArgs {
            args: Box::default(),
            has_self_type: false,
            bindings: Box::default(),
            parenthesized: GenericArgsParentheses::No,
        }
    }

    pub(crate) fn return_type_notation() -> GenericArgs {
        GenericArgs {
            args: Box::default(),
            has_self_type: false,
            bindings: Box::default(),
            parenthesized: GenericArgsParentheses::ReturnTypeNotation,
        }
    }
}

impl From<Name> for Path {
    fn from(name: Name) -> Path {
        Path::BarePath(Interned::new(ModPath::from_segments(PathKind::Plain, iter::once(name))))
    }
}
