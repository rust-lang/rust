//! A desugared representation of paths like `crate::foo` or `<Type as Trait>::bar`.
mod lower;
#[cfg(test)]
mod tests;

use std::{
    fmt::{self, Display},
    iter,
};

use crate::{
    lang_item::LangItemTarget,
    lower::LowerCtx,
    type_ref::{ConstRef, LifetimeRef, TypeBound, TypeRefId},
};
use hir_expand::name::Name;
use intern::Interned;
use span::Edition;
use stdx::thin_vec::thin_vec_with_header_struct;
use syntax::ast;

pub use hir_expand::mod_path::{path, ModPath, PathKind};

pub use lower::hir_segment_to_ast_segment;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportAlias {
    /// Unnamed alias, as in `use Foo as _;`
    Underscore,
    /// Named alias
    Alias(Name),
}

impl ImportAlias {
    pub fn display(&self, edition: Edition) -> impl Display + '_ {
        ImportAliasDisplay { value: self, edition }
    }
}

struct ImportAliasDisplay<'a> {
    value: &'a ImportAlias,
    edition: Edition,
}
impl Display for ImportAliasDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            ImportAlias::Underscore => f.write_str("_"),
            ImportAlias::Alias(name) => Display::fmt(&name.display_no_db(self.edition), f),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Path {
    /// `BarePath` is used when the path has neither generics nor type anchor, since the vast majority of paths
    /// are in this category, and splitting `Path` this way allows it to be more thin. When the path has either generics
    /// or type anchor, it is `Path::Normal` with the generics filled with `None` even if there are none (practically
    /// this is not a problem since many more paths have generics than a type anchor).
    BarePath(Interned<ModPath>),
    /// `Path::Normal` may have empty generics and type anchor (but generic args will be filled with `None`).
    Normal(NormalPath),
    /// A link to a lang item. It is used in desugaring of things like `it?`. We can show these
    /// links via a normal path since they might be private and not accessible in the usage place.
    LangItem(LangItemTarget, Option<Name>),
}

// This type is being used a lot, make sure it doesn't grow unintentionally.
#[cfg(target_arch = "x86_64")]
const _: () = {
    assert!(size_of::<Path>() == 16);
    assert!(size_of::<Option<Path>>() == 16);
};

thin_vec_with_header_struct! {
    pub new(pub(crate)) struct NormalPath, NormalPathHeader {
        pub generic_args: [Option<GenericArgs>],
        pub type_anchor: Option<TypeRefId>,
        pub mod_path: Interned<ModPath>; ref,
    }
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
    /// Whether these generic args were desugared from `Trait(Arg) -> Output`
    /// parenthesis notation typically used for the `Fn` traits.
    pub desugared_from_fn: bool,
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
    Lifetime(LifetimeRef),
    Const(ConstRef),
}

impl Path {
    /// Converts an `ast::Path` to `Path`. Works with use trees.
    /// It correctly handles `$crate` based path from macro call.
    pub fn from_src(ctx: &mut LowerCtx<'_>, path: ast::Path) -> Option<Path> {
        lower::lower_path(ctx, path)
    }

    /// Converts a known mod path to `Path`.
    pub fn from_known_path(path: ModPath, generic_args: Vec<Option<GenericArgs>>) -> Path {
        Path::Normal(NormalPath::new(None, Interned::new(path), generic_args))
    }

    /// Converts a known mod path to `Path`.
    pub fn from_known_path_with_no_generic(path: ModPath) -> Path {
        Path::BarePath(Interned::new(path))
    }

    #[inline]
    pub fn kind(&self) -> &PathKind {
        match self {
            Path::BarePath(mod_path) => &mod_path.kind,
            Path::Normal(path) => &path.mod_path().kind,
            Path::LangItem(..) => &PathKind::Abs,
        }
    }

    #[inline]
    pub fn type_anchor(&self) -> Option<TypeRefId> {
        match self {
            Path::Normal(path) => path.type_anchor(),
            Path::LangItem(..) | Path::BarePath(_) => None,
        }
    }

    #[inline]
    pub fn generic_args(&self) -> Option<&[Option<GenericArgs>]> {
        match self {
            Path::Normal(path) => Some(path.generic_args()),
            Path::LangItem(..) | Path::BarePath(_) => None,
        }
    }

    pub fn segments(&self) -> PathSegments<'_> {
        match self {
            Path::BarePath(mod_path) => {
                PathSegments { segments: mod_path.segments(), generic_args: None }
            }
            Path::Normal(path) => PathSegments {
                segments: path.mod_path().segments(),
                generic_args: Some(path.generic_args()),
            },
            Path::LangItem(_, seg) => PathSegments { segments: seg.as_slice(), generic_args: None },
        }
    }

    pub fn mod_path(&self) -> Option<&ModPath> {
        match self {
            Path::BarePath(mod_path) => Some(mod_path),
            Path::Normal(path) => Some(path.mod_path()),
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
                let mod_path = path.mod_path();
                if mod_path.is_ident() {
                    return None;
                }
                let type_anchor = path.type_anchor();
                let generic_args = path.generic_args();
                let qualifier_mod_path = Interned::new(ModPath::from_segments(
                    mod_path.kind,
                    mod_path.segments()[..mod_path.segments().len() - 1].iter().cloned(),
                ));
                let qualifier_generic_args = &generic_args[..generic_args.len() - 1];
                Some(Path::Normal(NormalPath::new(
                    type_anchor,
                    qualifier_mod_path,
                    qualifier_generic_args.iter().cloned(),
                )))
            }
            Path::LangItem(..) => None,
        }
    }

    pub fn is_self_type(&self) -> bool {
        match self {
            Path::BarePath(mod_path) => mod_path.is_Self(),
            Path::Normal(path) => {
                path.type_anchor().is_none()
                    && path.mod_path().is_Self()
                    && path.generic_args().iter().all(|args| args.is_none())
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
    pub(crate) fn from_ast(
        lower_ctx: &mut LowerCtx<'_>,
        node: ast::GenericArgList,
    ) -> Option<GenericArgs> {
        lower::lower_generic_args(lower_ctx, node)
    }

    pub(crate) fn empty() -> GenericArgs {
        GenericArgs {
            args: Box::default(),
            has_self_type: false,
            bindings: Box::default(),
            desugared_from_fn: false,
        }
    }
}

impl From<Name> for Path {
    fn from(name: Name) -> Path {
        Path::BarePath(Interned::new(ModPath::from_segments(PathKind::Plain, iter::once(name))))
    }
}
