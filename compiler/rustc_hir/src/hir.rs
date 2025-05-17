// ignore-tidy-filelength
use std::fmt;

use rustc_abi::ExternAbi;
use rustc_ast::attr::AttributeExt;
use rustc_ast::token::CommentKind;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_ast::{
    self as ast, FloatTy, InlineAsmOptions, InlineAsmTemplatePiece, IntTy, Label, LitIntType,
    LitKind, TraitObjectSyntax, UintTy, UnsafeBinderCastKind,
};
pub use rustc_ast::{
    AssignOp, AssignOpKind, AttrId, AttrStyle, BinOp, BinOpKind, BindingMode, BorrowKind,
    BoundConstness, BoundPolarity, ByRef, CaptureBy, DelimArgs, ImplPolarity, IsAuto,
    MetaItemInner, MetaItemLit, Movability, Mutability, UnOp,
};
use rustc_attr_data_structures::AttributeKind;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::tagged_ptr::TaggedRef;
use rustc_index::IndexVec;
use rustc_macros::{Decodable, Encodable, HashStable_Generic};
use rustc_span::def_id::LocalDefId;
use rustc_span::hygiene::MacroKind;
use rustc_span::source_map::Spanned;
use rustc_span::{BytePos, DUMMY_SP, ErrorGuaranteed, Ident, Span, Symbol, kw, sym};
use rustc_target::asm::InlineAsmRegOrRegClass;
use smallvec::SmallVec;
use thin_vec::ThinVec;
use tracing::debug;

use crate::LangItem;
use crate::def::{CtorKind, DefKind, Res};
use crate::def_id::{DefId, LocalDefIdMap};
pub(crate) use crate::hir_id::{HirId, ItemLocalId, ItemLocalMap, OwnerId};
use crate::intravisit::{FnKind, VisitorExt};

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable_Generic)]
pub enum AngleBrackets {
    /// E.g. `Path`.
    Missing,
    /// E.g. `Path<>`.
    Empty,
    /// E.g. `Path<T>`.
    Full,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable_Generic)]
pub enum LifetimeSource {
    /// E.g. `&Type`, `&'_ Type`, `&'a Type`, `&mut Type`, `&'_ mut Type`, `&'a mut Type`
    Reference,

    /// E.g. `ContainsLifetime`, `ContainsLifetime<>`, `ContainsLifetime<'_>`,
    /// `ContainsLifetime<'a>`
    Path { angle_brackets: AngleBrackets },

    /// E.g. `impl Trait + '_`, `impl Trait + 'a`
    OutlivesBound,

    /// E.g. `impl Trait + use<'_>`, `impl Trait + use<'a>`
    PreciseCapturing,

    /// Other usages which have not yet been categorized. Feel free to
    /// add new sources that you find useful.
    ///
    /// Some non-exhaustive examples:
    /// - `where T: 'a`
    /// - `fn(_: dyn Trait + 'a)`
    Other,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable_Generic)]
pub enum LifetimeSyntax {
    /// E.g. `&Type`, `ContainsLifetime`
    Hidden,

    /// E.g. `&'_ Type`, `ContainsLifetime<'_>`, `impl Trait + '_`, `impl Trait + use<'_>`
    Anonymous,

    /// E.g. `&'a Type`, `ContainsLifetime<'a>`, `impl Trait + 'a`, `impl Trait + use<'a>`
    Named,
}

impl From<Ident> for LifetimeSyntax {
    fn from(ident: Ident) -> Self {
        let name = ident.name;

        if name == kw::Empty {
            unreachable!("A lifetime name should never be empty");
        } else if name == kw::UnderscoreLifetime {
            LifetimeSyntax::Anonymous
        } else {
            debug_assert!(name.as_str().starts_with('\''));
            LifetimeSyntax::Named
        }
    }
}

/// A lifetime. The valid field combinations are non-obvious and not all
/// combinations are possible. The following example shows some of
/// them. See also the comments on `LifetimeKind` and `LifetimeSource`.
///
/// ```
/// #[repr(C)]
/// struct S<'a>(&'a u32);       // res=Param, name='a, source=Reference, syntax=Named
/// unsafe extern "C" {
///     fn f1(s: S);             // res=Param, name='_, source=Path, syntax=Hidden
///     fn f2(s: S<'_>);         // res=Param, name='_, source=Path, syntax=Anonymous
///     fn f3<'a>(s: S<'a>);     // res=Param, name='a, source=Path, syntax=Named
/// }
///
/// struct St<'a> { x: &'a u32 } // res=Param, name='a, source=Reference, syntax=Named
/// fn f() {
///     _ = St { x: &0 };        // res=Infer, name='_, source=Path, syntax=Hidden
///     _ = St::<'_> { x: &0 };  // res=Infer, name='_, source=Path, syntax=Anonymous
/// }
///
/// struct Name<'a>(&'a str);    // res=Param,  name='a, source=Reference, syntax=Named
/// const A: Name = Name("a");   // res=Static, name='_, source=Path, syntax=Hidden
/// const B: &str = "";          // res=Static, name='_, source=Reference, syntax=Hidden
/// static C: &'_ str = "";      // res=Static, name='_, source=Reference, syntax=Anonymous
/// static D: &'static str = ""; // res=Static, name='static, source=Reference, syntax=Named
///
/// trait Tr {}
/// fn tr(_: Box<dyn Tr>) {}     // res=ImplicitObjectLifetimeDefault, name='_, source=Other, syntax=Hidden
///
/// fn capture_outlives<'a>() ->
///     impl FnOnce() + 'a       // res=Param, ident='a, source=OutlivesBound, syntax=Named
/// {
///     || {}
/// }
///
/// fn capture_precise<'a>() ->
///     impl FnOnce() + use<'a>  // res=Param, ident='a, source=PreciseCapturing, syntax=Named
/// {
///     || {}
/// }
///
/// // (commented out because these cases trigger errors)
/// // struct S1<'a>(&'a str);   // res=Param, name='a, source=Reference, syntax=Named
/// // struct S2(S1);            // res=Error, name='_, source=Path, syntax=Hidden
/// // struct S3(S1<'_>);        // res=Error, name='_, source=Path, syntax=Anonymous
/// // struct S4(S1<'a>);        // res=Error, name='a, source=Path, syntax=Named
/// ```
///
/// Some combinations that cannot occur are `LifetimeSyntax::Hidden` with
/// `LifetimeSource::OutlivesBound` or `LifetimeSource::PreciseCapturing`
/// — there's no way to "elide" these lifetimes.
#[derive(Debug, Copy, Clone, HashStable_Generic)]
pub struct Lifetime {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,

    /// Either a named lifetime definition (e.g. `'a`, `'static`) or an
    /// anonymous lifetime (`'_`, either explicitly written, or inserted for
    /// things like `&type`).
    pub ident: Ident,

    /// Semantics of this lifetime.
    pub kind: LifetimeKind,

    /// The context in which the lifetime occurred. See `Lifetime::suggestion`
    /// for example use.
    pub source: LifetimeSource,

    /// The syntax that the user used to declare this lifetime. See
    /// `Lifetime::suggestion` for example use.
    pub syntax: LifetimeSyntax,
}

#[derive(Debug, Copy, Clone, HashStable_Generic)]
pub enum ParamName {
    /// Some user-given name like `T` or `'x`.
    Plain(Ident),

    /// Indicates an illegal name was given and an error has been
    /// reported (so we should squelch other derived errors).
    ///
    /// Occurs when, e.g., `'_` is used in the wrong place, or a
    /// lifetime name is duplicated.
    Error(Ident),

    /// Synthetic name generated when user elided a lifetime in an impl header.
    ///
    /// E.g., the lifetimes in cases like these:
    /// ```ignore (fragment)
    /// impl Foo for &u32
    /// impl Foo<'_> for u32
    /// ```
    /// in that case, we rewrite to
    /// ```ignore (fragment)
    /// impl<'f> Foo for &'f u32
    /// impl<'f> Foo<'f> for u32
    /// ```
    /// where `'f` is something like `Fresh(0)`. The indices are
    /// unique per impl, but not necessarily continuous.
    Fresh,
}

impl ParamName {
    pub fn ident(&self) -> Ident {
        match *self {
            ParamName::Plain(ident) | ParamName::Error(ident) => ident,
            ParamName::Fresh => Ident::with_dummy_span(kw::UnderscoreLifetime),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, HashStable_Generic)]
pub enum LifetimeKind {
    /// User-given names or fresh (synthetic) names.
    Param(LocalDefId),

    /// Implicit lifetime in a context like `dyn Foo`. This is
    /// distinguished from implicit lifetimes elsewhere because the
    /// lifetime that they default to must appear elsewhere within the
    /// enclosing type. This means that, in an `impl Trait` context, we
    /// don't have to create a parameter for them. That is, `impl
    /// Trait<Item = &u32>` expands to an opaque type like `type
    /// Foo<'a> = impl Trait<Item = &'a u32>`, but `impl Trait<item =
    /// dyn Bar>` expands to `type Foo = impl Trait<Item = dyn Bar +
    /// 'static>`. The latter uses `ImplicitObjectLifetimeDefault` so
    /// that surrounding code knows not to create a lifetime
    /// parameter.
    ImplicitObjectLifetimeDefault,

    /// Indicates an error during lowering (usually `'_` in wrong place)
    /// that was already reported.
    Error,

    /// User wrote an anonymous lifetime, either `'_` or nothing (which gets
    /// converted to `'_`). The semantics of this lifetime should be inferred
    /// by typechecking code.
    Infer,

    /// User wrote `'static` or nothing (which gets converted to `'_`).
    Static,
}

impl LifetimeKind {
    fn is_elided(&self) -> bool {
        match self {
            LifetimeKind::ImplicitObjectLifetimeDefault | LifetimeKind::Infer => true,

            // It might seem surprising that `Fresh` counts as not *elided*
            // -- but this is because, as far as the code in the compiler is
            // concerned -- `Fresh` variants act equivalently to "some fresh name".
            // They correspond to early-bound regions on an impl, in other words.
            LifetimeKind::Error | LifetimeKind::Param(..) | LifetimeKind::Static => false,
        }
    }
}

impl fmt::Display for Lifetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.ident.name.fmt(f)
    }
}

impl Lifetime {
    pub fn new(
        hir_id: HirId,
        ident: Ident,
        kind: LifetimeKind,
        source: LifetimeSource,
        syntax: LifetimeSyntax,
    ) -> Lifetime {
        let lifetime = Lifetime { hir_id, ident, kind, source, syntax };

        // Sanity check: elided lifetimes form a strict subset of anonymous lifetimes.
        #[cfg(debug_assertions)]
        match (lifetime.is_elided(), lifetime.is_anonymous()) {
            (false, false) => {} // e.g. `'a`
            (false, true) => {}  // e.g. explicit `'_`
            (true, true) => {}   // e.g. `&x`
            (true, false) => panic!("bad Lifetime"),
        }

        lifetime
    }

    pub fn is_elided(&self) -> bool {
        self.kind.is_elided()
    }

    pub fn is_anonymous(&self) -> bool {
        self.ident.name == kw::UnderscoreLifetime
    }

    pub fn is_syntactically_hidden(&self) -> bool {
        matches!(self.syntax, LifetimeSyntax::Hidden)
    }

    pub fn is_syntactically_anonymous(&self) -> bool {
        matches!(self.syntax, LifetimeSyntax::Anonymous)
    }

    pub fn is_static(&self) -> bool {
        self.kind == LifetimeKind::Static
    }

    pub fn suggestion(&self, new_lifetime: &str) -> (Span, String) {
        use LifetimeSource::*;
        use LifetimeSyntax::*;

        debug_assert!(new_lifetime.starts_with('\''));

        match (self.syntax, self.source) {
            // The user wrote `'a` or `'_`.
            (Named | Anonymous, _) => (self.ident.span, format!("{new_lifetime}")),

            // The user wrote `Path<T>`, and omitted the `'_,`.
            (Hidden, Path { angle_brackets: AngleBrackets::Full }) => {
                (self.ident.span, format!("{new_lifetime}, "))
            }

            // The user wrote `Path<>`, and omitted the `'_`..
            (Hidden, Path { angle_brackets: AngleBrackets::Empty }) => {
                (self.ident.span, format!("{new_lifetime}"))
            }

            // The user wrote `Path` and omitted the `<'_>`.
            (Hidden, Path { angle_brackets: AngleBrackets::Missing }) => {
                (self.ident.span.shrink_to_hi(), format!("<{new_lifetime}>"))
            }

            // The user wrote `&type` or `&mut type`.
            (Hidden, Reference) => (self.ident.span, format!("{new_lifetime} ")),

            (Hidden, source) => {
                unreachable!("can't suggest for a hidden lifetime of {source:?}")
            }
        }
    }
}

/// A `Path` is essentially Rust's notion of a name; for instance,
/// `std::cmp::PartialEq`. It's represented as a sequence of identifiers,
/// along with a bunch of supporting information.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Path<'hir, R = Res> {
    pub span: Span,
    /// The resolution for the path.
    pub res: R,
    /// The segments in the path: the things separated by `::`.
    pub segments: &'hir [PathSegment<'hir>],
}

/// Up to three resolutions for type, value and macro namespaces.
pub type UsePath<'hir> = Path<'hir, SmallVec<[Res; 3]>>;

impl Path<'_> {
    pub fn is_global(&self) -> bool {
        self.segments.first().is_some_and(|segment| segment.ident.name == kw::PathRoot)
    }
}

/// A segment of a path: an identifier, an optional lifetime, and a set of
/// types.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct PathSegment<'hir> {
    /// The identifier portion of this path segment.
    pub ident: Ident,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub res: Res,

    /// Type/lifetime parameters attached to this path. They come in
    /// two flavors: `Path<A,B,C>` and `Path(A,B) -> C`. Note that
    /// this is more than just simple syntactic sugar; the use of
    /// parens affects the region binding rules, so we preserve the
    /// distinction.
    pub args: Option<&'hir GenericArgs<'hir>>,

    /// Whether to infer remaining type parameters, if any.
    /// This only applies to expression and pattern paths, and
    /// out of those only the segments with no type parameters
    /// to begin with, e.g., `Vec::new` is `<Vec<..>>::new::<..>`.
    pub infer_args: bool,
}

impl<'hir> PathSegment<'hir> {
    /// Converts an identifier to the corresponding segment.
    pub fn new(ident: Ident, hir_id: HirId, res: Res) -> PathSegment<'hir> {
        PathSegment { ident, hir_id, res, infer_args: true, args: None }
    }

    pub fn invalid() -> Self {
        Self::new(Ident::dummy(), HirId::INVALID, Res::Err)
    }

    pub fn args(&self) -> &GenericArgs<'hir> {
        if let Some(ref args) = self.args {
            args
        } else {
            const DUMMY: &GenericArgs<'_> = &GenericArgs::none();
            DUMMY
        }
    }
}

/// A constant that enters the type system, used for arguments to const generics (e.g. array lengths).
///
/// These are distinct from [`AnonConst`] as anon consts in the type system are not allowed
/// to use any generic parameters, therefore we must represent `N` differently. Additionally
/// future designs for supporting generic parameters in const arguments will likely not use
/// an anon const based design.
///
/// So, `ConstArg` (specifically, [`ConstArgKind`]) distinguishes between const args
/// that are [just paths](ConstArgKind::Path) (currently just bare const params)
/// versus const args that are literals or have arbitrary computations (e.g., `{ 1 + 3 }`).
///
/// The `Unambig` generic parameter represents whether the position this const is from is
/// unambiguously a const or ambiguous as to whether it is a type or a const. When in an
/// ambiguous context the parameter is instantiated with an uninhabited type making the
/// [`ConstArgKind::Infer`] variant unusable and [`GenericArg::Infer`] is used instead.
#[derive(Clone, Copy, Debug, HashStable_Generic)]
#[repr(C)]
pub struct ConstArg<'hir, Unambig = ()> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: ConstArgKind<'hir, Unambig>,
}

impl<'hir> ConstArg<'hir, AmbigArg> {
    /// Converts a `ConstArg` in an ambiguous position to one in an unambiguous position.
    ///
    /// Functions accepting an unambiguous consts may expect the [`ConstArgKind::Infer`] variant
    /// to be used. Care should be taken to separately handle infer consts when calling this
    /// function as it cannot be handled by downstream code making use of the returned const.
    ///
    /// In practice this may mean overriding the [`Visitor::visit_infer`][visit_infer] method on hir visitors, or
    /// specifically matching on [`GenericArg::Infer`] when handling generic arguments.
    ///
    /// [visit_infer]: [rustc_hir::intravisit::Visitor::visit_infer]
    pub fn as_unambig_ct(&self) -> &ConstArg<'hir> {
        // SAFETY: `ConstArg` is `repr(C)` and `ConstArgKind` is marked `repr(u8)` so that the
        // layout is the same across different ZST type arguments.
        let ptr = self as *const ConstArg<'hir, AmbigArg> as *const ConstArg<'hir, ()>;
        unsafe { &*ptr }
    }
}

impl<'hir> ConstArg<'hir> {
    /// Converts a `ConstArg` in an unambigous position to one in an ambiguous position. This is
    /// fallible as the [`ConstArgKind::Infer`] variant is not present in ambiguous positions.
    ///
    /// Functions accepting ambiguous consts will not handle the [`ConstArgKind::Infer`] variant, if
    /// infer consts are relevant to you then care should be taken to handle them separately.
    pub fn try_as_ambig_ct(&self) -> Option<&ConstArg<'hir, AmbigArg>> {
        if let ConstArgKind::Infer(_, ()) = self.kind {
            return None;
        }

        // SAFETY: `ConstArg` is `repr(C)` and `ConstArgKind` is marked `repr(u8)` so that the layout is
        // the same across different ZST type arguments. We also asserted that the `self` is
        // not a `ConstArgKind::Infer` so there is no risk of transmuting a `()` to `AmbigArg`.
        let ptr = self as *const ConstArg<'hir> as *const ConstArg<'hir, AmbigArg>;
        Some(unsafe { &*ptr })
    }
}

impl<'hir, Unambig> ConstArg<'hir, Unambig> {
    pub fn anon_const_hir_id(&self) -> Option<HirId> {
        match self.kind {
            ConstArgKind::Anon(ac) => Some(ac.hir_id),
            _ => None,
        }
    }

    pub fn span(&self) -> Span {
        match self.kind {
            ConstArgKind::Path(path) => path.span(),
            ConstArgKind::Anon(anon) => anon.span,
            ConstArgKind::Infer(span, _) => span,
        }
    }
}

/// See [`ConstArg`].
#[derive(Clone, Copy, Debug, HashStable_Generic)]
#[repr(u8, C)]
pub enum ConstArgKind<'hir, Unambig = ()> {
    /// **Note:** Currently this is only used for bare const params
    /// (`N` where `fn foo<const N: usize>(...)`),
    /// not paths to any const (`N` where `const N: usize = ...`).
    ///
    /// However, in the future, we'll be using it for all of those.
    Path(QPath<'hir>),
    Anon(&'hir AnonConst),
    /// This variant is not always used to represent inference consts, sometimes
    /// [`GenericArg::Infer`] is used instead.
    Infer(Span, Unambig),
}

#[derive(Clone, Copy, Debug, HashStable_Generic)]
pub struct InferArg {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
}

impl InferArg {
    pub fn to_ty(&self) -> Ty<'static> {
        Ty { kind: TyKind::Infer(()), span: self.span, hir_id: self.hir_id }
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum GenericArg<'hir> {
    Lifetime(&'hir Lifetime),
    Type(&'hir Ty<'hir, AmbigArg>),
    Const(&'hir ConstArg<'hir, AmbigArg>),
    /// Inference variables in [`GenericArg`] are always represnted by
    /// `GenericArg::Infer` instead of the `Infer` variants on [`TyKind`] and
    /// [`ConstArgKind`] as it is not clear until hir ty lowering whether a
    /// `_` argument is a type or const argument.
    ///
    /// However, some builtin types' generic arguments are represented by [`TyKind`]
    /// without a [`GenericArg`], instead directly storing a [`Ty`] or [`ConstArg`]. In
    /// such cases they *are* represented by the `Infer` variants on [`TyKind`] and
    /// [`ConstArgKind`] as it is not ambiguous whether the argument is a type or const.
    Infer(InferArg),
}

impl GenericArg<'_> {
    pub fn span(&self) -> Span {
        match self {
            GenericArg::Lifetime(l) => l.ident.span,
            GenericArg::Type(t) => t.span,
            GenericArg::Const(c) => c.span(),
            GenericArg::Infer(i) => i.span,
        }
    }

    pub fn hir_id(&self) -> HirId {
        match self {
            GenericArg::Lifetime(l) => l.hir_id,
            GenericArg::Type(t) => t.hir_id,
            GenericArg::Const(c) => c.hir_id,
            GenericArg::Infer(i) => i.hir_id,
        }
    }

    pub fn descr(&self) -> &'static str {
        match self {
            GenericArg::Lifetime(_) => "lifetime",
            GenericArg::Type(_) => "type",
            GenericArg::Const(_) => "constant",
            GenericArg::Infer(_) => "placeholder",
        }
    }

    pub fn to_ord(&self) -> ast::ParamKindOrd {
        match self {
            GenericArg::Lifetime(_) => ast::ParamKindOrd::Lifetime,
            GenericArg::Type(_) | GenericArg::Const(_) | GenericArg::Infer(_) => {
                ast::ParamKindOrd::TypeOrConst
            }
        }
    }

    pub fn is_ty_or_const(&self) -> bool {
        match self {
            GenericArg::Lifetime(_) => false,
            GenericArg::Type(_) | GenericArg::Const(_) | GenericArg::Infer(_) => true,
        }
    }
}

/// The generic arguments and associated item constraints of a path segment.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct GenericArgs<'hir> {
    /// The generic arguments for this path segment.
    pub args: &'hir [GenericArg<'hir>],
    /// The associated item constraints for this path segment.
    pub constraints: &'hir [AssocItemConstraint<'hir>],
    /// Whether the arguments were written in parenthesized form (e.g., `Fn(T) -> U`).
    ///
    /// This is required mostly for pretty-printing and diagnostics,
    /// but also for changing lifetime elision rules to be "function-like".
    pub parenthesized: GenericArgsParentheses,
    /// The span encompassing the arguments, constraints and the surrounding brackets (`<>` or `()`).
    ///
    /// For example:
    ///
    /// ```ignore (illustrative)
    ///       Foo<A, B, AssocTy = D>           Fn(T, U, V) -> W
    ///          ^^^^^^^^^^^^^^^^^^^             ^^^^^^^^^
    /// ```
    ///
    /// Note that this may be:
    /// - empty, if there are no generic brackets (but there may be hidden lifetimes)
    /// - dummy, if this was generated during desugaring
    pub span_ext: Span,
}

impl<'hir> GenericArgs<'hir> {
    pub const fn none() -> Self {
        Self {
            args: &[],
            constraints: &[],
            parenthesized: GenericArgsParentheses::No,
            span_ext: DUMMY_SP,
        }
    }

    /// Obtain the list of input types and the output type if the generic arguments are parenthesized.
    ///
    /// Returns the `Ty0, Ty1, ...` and the `RetTy` in `Trait(Ty0, Ty1, ...) -> RetTy`.
    /// Panics if the parenthesized arguments have an incorrect form (this shouldn't happen).
    pub fn paren_sugar_inputs_output(&self) -> Option<(&[Ty<'hir>], &Ty<'hir>)> {
        if self.parenthesized != GenericArgsParentheses::ParenSugar {
            return None;
        }

        let inputs = self
            .args
            .iter()
            .find_map(|arg| {
                let GenericArg::Type(ty) = arg else { return None };
                let TyKind::Tup(tys) = &ty.kind else { return None };
                Some(tys)
            })
            .unwrap();

        Some((inputs, self.paren_sugar_output_inner()))
    }

    /// Obtain the output type if the generic arguments are parenthesized.
    ///
    /// Returns the `RetTy` in `Trait(Ty0, Ty1, ...) -> RetTy`.
    /// Panics if the parenthesized arguments have an incorrect form (this shouldn't happen).
    pub fn paren_sugar_output(&self) -> Option<&Ty<'hir>> {
        (self.parenthesized == GenericArgsParentheses::ParenSugar)
            .then(|| self.paren_sugar_output_inner())
    }

    fn paren_sugar_output_inner(&self) -> &Ty<'hir> {
        let [constraint] = self.constraints.try_into().unwrap();
        debug_assert_eq!(constraint.ident.name, sym::Output);
        constraint.ty().unwrap()
    }

    pub fn has_err(&self) -> Option<ErrorGuaranteed> {
        self.args
            .iter()
            .find_map(|arg| {
                let GenericArg::Type(ty) = arg else { return None };
                let TyKind::Err(guar) = ty.kind else { return None };
                Some(guar)
            })
            .or_else(|| {
                self.constraints.iter().find_map(|constraint| {
                    let TyKind::Err(guar) = constraint.ty()?.kind else { return None };
                    Some(guar)
                })
            })
    }

    #[inline]
    pub fn num_lifetime_params(&self) -> usize {
        self.args.iter().filter(|arg| matches!(arg, GenericArg::Lifetime(_))).count()
    }

    #[inline]
    pub fn has_lifetime_params(&self) -> bool {
        self.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
    }

    #[inline]
    /// This function returns the number of type and const generic params.
    /// It should only be used for diagnostics.
    pub fn num_generic_params(&self) -> usize {
        self.args.iter().filter(|arg| !matches!(arg, GenericArg::Lifetime(_))).count()
    }

    /// The span encompassing the arguments and constraints[^1] inside the surrounding brackets.
    ///
    /// Returns `None` if the span is empty (i.e., no brackets) or dummy.
    ///
    /// [^1]: Unless of the form `-> Ty` (see [`GenericArgsParentheses`]).
    pub fn span(&self) -> Option<Span> {
        let span_ext = self.span_ext()?;
        Some(span_ext.with_lo(span_ext.lo() + BytePos(1)).with_hi(span_ext.hi() - BytePos(1)))
    }

    /// Returns span encompassing arguments and their surrounding `<>` or `()`
    pub fn span_ext(&self) -> Option<Span> {
        Some(self.span_ext).filter(|span| !span.is_empty())
    }

    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, HashStable_Generic)]
pub enum GenericArgsParentheses {
    No,
    /// Bounds for `feature(return_type_notation)`, like `T: Trait<method(..): Send>`,
    /// where the args are explicitly elided with `..`
    ReturnTypeNotation,
    /// parenthesized function-family traits, like `T: Fn(u32) -> i32`
    ParenSugar,
}

/// The modifiers on a trait bound.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct TraitBoundModifiers {
    pub constness: BoundConstness,
    pub polarity: BoundPolarity,
}

impl TraitBoundModifiers {
    pub const NONE: Self =
        TraitBoundModifiers { constness: BoundConstness::Never, polarity: BoundPolarity::Positive };
}

#[derive(Clone, Copy, Debug, HashStable_Generic)]
pub enum GenericBound<'hir> {
    Trait(PolyTraitRef<'hir>),
    Outlives(&'hir Lifetime),
    Use(&'hir [PreciseCapturingArg<'hir>], Span),
}

impl GenericBound<'_> {
    pub fn trait_ref(&self) -> Option<&TraitRef<'_>> {
        match self {
            GenericBound::Trait(data) => Some(&data.trait_ref),
            _ => None,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            GenericBound::Trait(t, ..) => t.span,
            GenericBound::Outlives(l) => l.ident.span,
            GenericBound::Use(_, span) => *span,
        }
    }
}

pub type GenericBounds<'hir> = &'hir [GenericBound<'hir>];

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable_Generic, Debug)]
pub enum MissingLifetimeKind {
    /// An explicit `'_`.
    Underscore,
    /// An elided lifetime `&' ty`.
    Ampersand,
    /// An elided lifetime in brackets with written brackets.
    Comma,
    /// An elided lifetime with elided brackets.
    Brackets,
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum LifetimeParamKind {
    // Indicates that the lifetime definition was explicitly declared (e.g., in
    // `fn foo<'a>(x: &'a u8) -> &'a u8 { x }`).
    Explicit,

    // Indication that the lifetime was elided (e.g., in both cases in
    // `fn foo(x: &u8) -> &'_ u8 { x }`).
    Elided(MissingLifetimeKind),

    // Indication that the lifetime name was somehow in error.
    Error,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum GenericParamKind<'hir> {
    /// A lifetime definition (e.g., `'a: 'b + 'c + 'd`).
    Lifetime {
        kind: LifetimeParamKind,
    },
    Type {
        default: Option<&'hir Ty<'hir>>,
        synthetic: bool,
    },
    Const {
        ty: &'hir Ty<'hir>,
        /// Optional default value for the const generic param
        default: Option<&'hir ConstArg<'hir>>,
        synthetic: bool,
    },
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct GenericParam<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    pub name: ParamName,
    pub span: Span,
    pub pure_wrt_drop: bool,
    pub kind: GenericParamKind<'hir>,
    pub colon_span: Option<Span>,
    pub source: GenericParamSource,
}

impl<'hir> GenericParam<'hir> {
    /// Synthetic type-parameters are inserted after normal ones.
    /// In order for normal parameters to be able to refer to synthetic ones,
    /// scans them first.
    pub fn is_impl_trait(&self) -> bool {
        matches!(self.kind, GenericParamKind::Type { synthetic: true, .. })
    }

    /// This can happen for `async fn`, e.g. `async fn f<'_>(&'_ self)`.
    ///
    /// See `lifetime_to_generic_param` in `rustc_ast_lowering` for more information.
    pub fn is_elided_lifetime(&self) -> bool {
        matches!(self.kind, GenericParamKind::Lifetime { kind: LifetimeParamKind::Elided(_) })
    }
}

/// Records where the generic parameter originated from.
///
/// This can either be from an item's generics, in which case it's typically
/// early-bound (but can be a late-bound lifetime in functions, for example),
/// or from a `for<...>` binder, in which case it's late-bound (and notably,
/// does not show up in the parent item's generics).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum GenericParamSource {
    // Early or late-bound parameters defined on an item
    Generics,
    // Late-bound parameters defined via a `for<...>`
    Binder,
}

#[derive(Default)]
pub struct GenericParamCount {
    pub lifetimes: usize,
    pub types: usize,
    pub consts: usize,
    pub infer: usize,
}

/// Represents lifetimes and type parameters attached to a declaration
/// of a function, enum, trait, etc.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Generics<'hir> {
    pub params: &'hir [GenericParam<'hir>],
    pub predicates: &'hir [WherePredicate<'hir>],
    pub has_where_clause_predicates: bool,
    pub where_clause_span: Span,
    pub span: Span,
}

impl<'hir> Generics<'hir> {
    pub const fn empty() -> &'hir Generics<'hir> {
        const NOPE: Generics<'_> = Generics {
            params: &[],
            predicates: &[],
            has_where_clause_predicates: false,
            where_clause_span: DUMMY_SP,
            span: DUMMY_SP,
        };
        &NOPE
    }

    pub fn get_named(&self, name: Symbol) -> Option<&GenericParam<'hir>> {
        self.params.iter().find(|&param| name == param.name.ident().name)
    }

    /// If there are generic parameters, return where to introduce a new one.
    pub fn span_for_lifetime_suggestion(&self) -> Option<Span> {
        if let Some(first) = self.params.first()
            && self.span.contains(first.span)
        {
            // `fn foo<A>(t: impl Trait)`
            //         ^ suggest `'a, ` here
            Some(first.span.shrink_to_lo())
        } else {
            None
        }
    }

    /// If there are generic parameters, return where to introduce a new one.
    pub fn span_for_param_suggestion(&self) -> Option<Span> {
        self.params.iter().any(|p| self.span.contains(p.span)).then(|| {
            // `fn foo<A>(t: impl Trait)`
            //          ^ suggest `, T: Trait` here
            self.span.with_lo(self.span.hi() - BytePos(1)).shrink_to_lo()
        })
    }

    /// `Span` where further predicates would be suggested, accounting for trailing commas, like
    ///  in `fn foo<T>(t: T) where T: Foo,` so we don't suggest two trailing commas.
    pub fn tail_span_for_predicate_suggestion(&self) -> Span {
        let end = self.where_clause_span.shrink_to_hi();
        if self.has_where_clause_predicates {
            self.predicates
                .iter()
                .rfind(|&p| p.kind.in_where_clause())
                .map_or(end, |p| p.span)
                .shrink_to_hi()
                .to(end)
        } else {
            end
        }
    }

    pub fn add_where_or_trailing_comma(&self) -> &'static str {
        if self.has_where_clause_predicates {
            ","
        } else if self.where_clause_span.is_empty() {
            " where"
        } else {
            // No where clause predicates, but we have `where` token
            ""
        }
    }

    pub fn bounds_for_param(
        &self,
        param_def_id: LocalDefId,
    ) -> impl Iterator<Item = &WhereBoundPredicate<'hir>> {
        self.predicates.iter().filter_map(move |pred| match pred.kind {
            WherePredicateKind::BoundPredicate(bp)
                if bp.is_param_bound(param_def_id.to_def_id()) =>
            {
                Some(bp)
            }
            _ => None,
        })
    }

    pub fn outlives_for_param(
        &self,
        param_def_id: LocalDefId,
    ) -> impl Iterator<Item = &WhereRegionPredicate<'_>> {
        self.predicates.iter().filter_map(move |pred| match pred.kind {
            WherePredicateKind::RegionPredicate(rp) if rp.is_param_bound(param_def_id) => Some(rp),
            _ => None,
        })
    }

    /// Returns a suggestable empty span right after the "final" bound of the generic parameter.
    ///
    /// If that bound needs to be wrapped in parentheses to avoid ambiguity with
    /// subsequent bounds, it also returns an empty span for an open parenthesis
    /// as the second component.
    ///
    /// E.g., adding `+ 'static` after `Fn() -> dyn Future<Output = ()>` or
    /// `Fn() -> &'static dyn Debug` requires parentheses:
    /// `Fn() -> (dyn Future<Output = ()>) + 'static` and
    /// `Fn() -> &'static (dyn Debug) + 'static`, respectively.
    pub fn bounds_span_for_suggestions(
        &self,
        param_def_id: LocalDefId,
    ) -> Option<(Span, Option<Span>)> {
        self.bounds_for_param(param_def_id).flat_map(|bp| bp.bounds.iter().rev()).find_map(
            |bound| {
                let span_for_parentheses = if let Some(trait_ref) = bound.trait_ref()
                    && let [.., segment] = trait_ref.path.segments
                    && let Some(ret_ty) = segment.args().paren_sugar_output()
                    && let ret_ty = ret_ty.peel_refs()
                    && let TyKind::TraitObject(_, tagged_ptr) = ret_ty.kind
                    && let TraitObjectSyntax::Dyn | TraitObjectSyntax::DynStar = tagged_ptr.tag()
                    && ret_ty.span.can_be_used_for_suggestions()
                {
                    Some(ret_ty.span)
                } else {
                    None
                };

                span_for_parentheses.map_or_else(
                    || {
                        // We include bounds that come from a `#[derive(_)]` but point at the user's code,
                        // as we use this method to get a span appropriate for suggestions.
                        let bs = bound.span();
                        bs.can_be_used_for_suggestions().then(|| (bs.shrink_to_hi(), None))
                    },
                    |span| Some((span.shrink_to_hi(), Some(span.shrink_to_lo()))),
                )
            },
        )
    }

    pub fn span_for_predicate_removal(&self, pos: usize) -> Span {
        let predicate = &self.predicates[pos];
        let span = predicate.span;

        if !predicate.kind.in_where_clause() {
            // <T: ?Sized, U>
            //   ^^^^^^^^
            return span;
        }

        // We need to find out which comma to remove.
        if pos < self.predicates.len() - 1 {
            let next_pred = &self.predicates[pos + 1];
            if next_pred.kind.in_where_clause() {
                // where T: ?Sized, Foo: Bar,
                //       ^^^^^^^^^^^
                return span.until(next_pred.span);
            }
        }

        if pos > 0 {
            let prev_pred = &self.predicates[pos - 1];
            if prev_pred.kind.in_where_clause() {
                // where Foo: Bar, T: ?Sized,
                //               ^^^^^^^^^^^
                return prev_pred.span.shrink_to_hi().to(span);
            }
        }

        // This is the only predicate in the where clause.
        // where T: ?Sized
        // ^^^^^^^^^^^^^^^
        self.where_clause_span
    }

    pub fn span_for_bound_removal(&self, predicate_pos: usize, bound_pos: usize) -> Span {
        let predicate = &self.predicates[predicate_pos];
        let bounds = predicate.kind.bounds();

        if bounds.len() == 1 {
            return self.span_for_predicate_removal(predicate_pos);
        }

        let bound_span = bounds[bound_pos].span();
        if bound_pos < bounds.len() - 1 {
            // If there's another bound after the current bound
            // include the following '+' e.g.:
            //
            //  `T: Foo + CurrentBound + Bar`
            //            ^^^^^^^^^^^^^^^
            bound_span.to(bounds[bound_pos + 1].span().shrink_to_lo())
        } else {
            // If the current bound is the last bound
            // include the preceding '+' E.g.:
            //
            //  `T: Foo + Bar + CurrentBound`
            //               ^^^^^^^^^^^^^^^
            bound_span.with_lo(bounds[bound_pos - 1].span().hi())
        }
    }
}

/// A single predicate in a where-clause.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct WherePredicate<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    pub kind: &'hir WherePredicateKind<'hir>,
}

/// The kind of a single predicate in a where-clause.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum WherePredicateKind<'hir> {
    /// A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
    BoundPredicate(WhereBoundPredicate<'hir>),
    /// A lifetime predicate (e.g., `'a: 'b + 'c`).
    RegionPredicate(WhereRegionPredicate<'hir>),
    /// An equality predicate (unsupported).
    EqPredicate(WhereEqPredicate<'hir>),
}

impl<'hir> WherePredicateKind<'hir> {
    pub fn in_where_clause(&self) -> bool {
        match self {
            WherePredicateKind::BoundPredicate(p) => p.origin == PredicateOrigin::WhereClause,
            WherePredicateKind::RegionPredicate(p) => p.in_where_clause,
            WherePredicateKind::EqPredicate(_) => false,
        }
    }

    pub fn bounds(&self) -> GenericBounds<'hir> {
        match self {
            WherePredicateKind::BoundPredicate(p) => p.bounds,
            WherePredicateKind::RegionPredicate(p) => p.bounds,
            WherePredicateKind::EqPredicate(_) => &[],
        }
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic, PartialEq, Eq)]
pub enum PredicateOrigin {
    WhereClause,
    GenericParam,
    ImplTrait,
}

/// A type bound (e.g., `for<'c> Foo: Send + Clone + 'c`).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct WhereBoundPredicate<'hir> {
    /// Origin of the predicate.
    pub origin: PredicateOrigin,
    /// Any generics from a `for` binding.
    pub bound_generic_params: &'hir [GenericParam<'hir>],
    /// The type being bounded.
    pub bounded_ty: &'hir Ty<'hir>,
    /// Trait and lifetime bounds (e.g., `Clone + Send + 'static`).
    pub bounds: GenericBounds<'hir>,
}

impl<'hir> WhereBoundPredicate<'hir> {
    /// Returns `true` if `param_def_id` matches the `bounded_ty` of this predicate.
    pub fn is_param_bound(&self, param_def_id: DefId) -> bool {
        self.bounded_ty.as_generic_param().is_some_and(|(def_id, _)| def_id == param_def_id)
    }
}

/// A lifetime predicate (e.g., `'a: 'b + 'c`).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct WhereRegionPredicate<'hir> {
    pub in_where_clause: bool,
    pub lifetime: &'hir Lifetime,
    pub bounds: GenericBounds<'hir>,
}

impl<'hir> WhereRegionPredicate<'hir> {
    /// Returns `true` if `param_def_id` matches the `lifetime` of this predicate.
    fn is_param_bound(&self, param_def_id: LocalDefId) -> bool {
        self.lifetime.kind == LifetimeKind::Param(param_def_id)
    }
}

/// An equality predicate (e.g., `T = int`); currently unsupported.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct WhereEqPredicate<'hir> {
    pub lhs_ty: &'hir Ty<'hir>,
    pub rhs_ty: &'hir Ty<'hir>,
}

/// HIR node coupled with its parent's id in the same HIR owner.
///
/// The parent is trash when the node is a HIR owner.
#[derive(Clone, Copy, Debug)]
pub struct ParentedNode<'tcx> {
    pub parent: ItemLocalId,
    pub node: Node<'tcx>,
}

/// Arguments passed to an attribute macro.
#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum AttrArgs {
    /// No arguments: `#[attr]`.
    Empty,
    /// Delimited arguments: `#[attr()/[]/{}]`.
    Delimited(DelimArgs),
    /// Arguments of a key-value attribute: `#[attr = "value"]`.
    Eq {
        /// Span of the `=` token.
        eq_span: Span,
        /// The "value".
        expr: MetaItemLit,
    },
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct AttrPath {
    pub segments: Box<[Ident]>,
    pub span: Span,
}

impl AttrPath {
    pub fn from_ast(path: &ast::Path) -> Self {
        AttrPath {
            segments: path.segments.iter().map(|i| i.ident).collect::<Vec<_>>().into_boxed_slice(),
            span: path.span,
        }
    }
}

impl fmt::Display for AttrPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.segments.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("::"))
    }
}

#[derive(Clone, Debug, HashStable_Generic, Encodable, Decodable)]
pub struct AttrItem {
    // Not lowered to hir::Path because we have no NodeId to resolve to.
    pub path: AttrPath,
    pub args: AttrArgs,
    pub id: HashIgnoredAttrId,
    /// Denotes if the attribute decorates the following construct (outer)
    /// or the construct this attribute is contained within (inner).
    pub style: AttrStyle,
    /// Span of the entire attribute
    pub span: Span,
}

/// The derived implementation of [`HashStable_Generic`] on [`Attribute`]s shouldn't hash
/// [`AttrId`]s. By wrapping them in this, we make sure we never do.
#[derive(Copy, Debug, Encodable, Decodable, Clone)]
pub struct HashIgnoredAttrId {
    pub attr_id: AttrId,
}

#[derive(Clone, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Attribute {
    /// A parsed built-in attribute.
    ///
    /// Each attribute has a span connected to it. However, you must be somewhat careful using it.
    /// That's because sometimes we merge multiple attributes together, like when an item has
    /// multiple `repr` attributes. In this case the span might not be very useful.
    Parsed(AttributeKind),

    /// An attribute that could not be parsed, out of a token-like representation.
    /// This is the case for custom tool attributes.
    Unparsed(Box<AttrItem>),
}

impl Attribute {
    pub fn get_normal_item(&self) -> &AttrItem {
        match &self {
            Attribute::Unparsed(normal) => &normal,
            _ => panic!("unexpected parsed attribute"),
        }
    }

    pub fn unwrap_normal_item(self) -> AttrItem {
        match self {
            Attribute::Unparsed(normal) => *normal,
            _ => panic!("unexpected parsed attribute"),
        }
    }

    pub fn value_lit(&self) -> Option<&MetaItemLit> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Eq { eq_span: _, expr }, .. } => Some(expr),
                _ => None,
            },
            _ => None,
        }
    }
}

impl AttributeExt for Attribute {
    #[inline]
    fn id(&self) -> AttrId {
        match &self {
            Attribute::Unparsed(u) => u.id.attr_id,
            _ => panic!(),
        }
    }

    #[inline]
    fn meta_item_list(&self) -> Option<ThinVec<ast::MetaItemInner>> {
        match &self {
            Attribute::Unparsed(n) => match n.as_ref() {
                AttrItem { args: AttrArgs::Delimited(d), .. } => {
                    ast::MetaItemKind::list_from_tokens(d.tokens.clone())
                }
                _ => None,
            },
            _ => None,
        }
    }

    #[inline]
    fn value_str(&self) -> Option<Symbol> {
        self.value_lit().and_then(|x| x.value_str())
    }

    #[inline]
    fn value_span(&self) -> Option<Span> {
        self.value_lit().map(|i| i.span)
    }

    /// For a single-segment attribute, returns its name; otherwise, returns `None`.
    #[inline]
    fn ident(&self) -> Option<Ident> {
        match &self {
            Attribute::Unparsed(n) => {
                if let [ident] = n.path.segments.as_ref() {
                    Some(*ident)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn path_matches(&self, name: &[Symbol]) -> bool {
        match &self {
            Attribute::Unparsed(n) => {
                n.path.segments.len() == name.len()
                    && n.path.segments.iter().zip(name).all(|(s, n)| s.name == *n)
            }
            _ => false,
        }
    }

    #[inline]
    fn is_doc_comment(&self) -> bool {
        matches!(self, Attribute::Parsed(AttributeKind::DocComment { .. }))
    }

    #[inline]
    fn span(&self) -> Span {
        match &self {
            Attribute::Unparsed(u) => u.span,
            // FIXME: should not be needed anymore when all attrs are parsed
            Attribute::Parsed(AttributeKind::Deprecation { span, .. }) => *span,
            Attribute::Parsed(AttributeKind::DocComment { span, .. }) => *span,
            a => panic!("can't get the span of an arbitrary parsed attribute: {a:?}"),
        }
    }

    #[inline]
    fn is_word(&self) -> bool {
        match &self {
            Attribute::Unparsed(n) => {
                matches!(n.args, AttrArgs::Empty)
            }
            _ => false,
        }
    }

    #[inline]
    fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        match &self {
            Attribute::Unparsed(n) => Some(n.path.segments.iter().copied().collect()),
            _ => None,
        }
    }

    #[inline]
    fn doc_str(&self) -> Option<Symbol> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { comment, .. }) => Some(*comment),
            Attribute::Unparsed(_) if self.has_name(sym::doc) => self.value_str(),
            _ => None,
        }
    }
    #[inline]
    fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        match &self {
            Attribute::Parsed(AttributeKind::DocComment { kind, comment, .. }) => {
                Some((*comment, *kind))
            }
            Attribute::Unparsed(_) if self.has_name(sym::doc) => {
                self.value_str().map(|s| (s, CommentKind::Line))
            }
            _ => None,
        }
    }

    #[inline]
    fn style(&self) -> AttrStyle {
        match &self {
            Attribute::Unparsed(u) => u.style,
            Attribute::Parsed(AttributeKind::DocComment { style, .. }) => *style,
            _ => panic!(),
        }
    }
}

// FIXME(fn_delegation): use function delegation instead of manually forwarding
impl Attribute {
    #[inline]
    pub fn id(&self) -> AttrId {
        AttributeExt::id(self)
    }

    #[inline]
    pub fn name(&self) -> Option<Symbol> {
        AttributeExt::name(self)
    }

    #[inline]
    pub fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        AttributeExt::meta_item_list(self)
    }

    #[inline]
    pub fn value_str(&self) -> Option<Symbol> {
        AttributeExt::value_str(self)
    }

    #[inline]
    pub fn value_span(&self) -> Option<Span> {
        AttributeExt::value_span(self)
    }

    #[inline]
    pub fn ident(&self) -> Option<Ident> {
        AttributeExt::ident(self)
    }

    #[inline]
    pub fn path_matches(&self, name: &[Symbol]) -> bool {
        AttributeExt::path_matches(self, name)
    }

    #[inline]
    pub fn is_doc_comment(&self) -> bool {
        AttributeExt::is_doc_comment(self)
    }

    #[inline]
    pub fn has_name(&self, name: Symbol) -> bool {
        AttributeExt::has_name(self, name)
    }

    #[inline]
    pub fn has_any_name(&self, names: &[Symbol]) -> bool {
        AttributeExt::has_any_name(self, names)
    }

    #[inline]
    pub fn span(&self) -> Span {
        AttributeExt::span(self)
    }

    #[inline]
    pub fn is_word(&self) -> bool {
        AttributeExt::is_word(self)
    }

    #[inline]
    pub fn path(&self) -> SmallVec<[Symbol; 1]> {
        AttributeExt::path(self)
    }

    #[inline]
    pub fn ident_path(&self) -> Option<SmallVec<[Ident; 1]>> {
        AttributeExt::ident_path(self)
    }

    #[inline]
    pub fn doc_str(&self) -> Option<Symbol> {
        AttributeExt::doc_str(self)
    }

    #[inline]
    pub fn is_proc_macro_attr(&self) -> bool {
        AttributeExt::is_proc_macro_attr(self)
    }

    #[inline]
    pub fn doc_str_and_comment_kind(&self) -> Option<(Symbol, CommentKind)> {
        AttributeExt::doc_str_and_comment_kind(self)
    }

    #[inline]
    pub fn style(&self) -> AttrStyle {
        AttributeExt::style(self)
    }
}

/// Attributes owned by a HIR owner.
#[derive(Debug)]
pub struct AttributeMap<'tcx> {
    pub map: SortedMap<ItemLocalId, &'tcx [Attribute]>,
    /// Preprocessed `#[define_opaque]` attribute.
    pub define_opaque: Option<&'tcx [(Span, LocalDefId)]>,
    // Only present when the crate hash is needed.
    pub opt_hash: Option<Fingerprint>,
}

impl<'tcx> AttributeMap<'tcx> {
    pub const EMPTY: &'static AttributeMap<'static> = &AttributeMap {
        map: SortedMap::new(),
        opt_hash: Some(Fingerprint::ZERO),
        define_opaque: None,
    };

    #[inline]
    pub fn get(&self, id: ItemLocalId) -> &'tcx [Attribute] {
        self.map.get(&id).copied().unwrap_or(&[])
    }
}

/// Map of all HIR nodes inside the current owner.
/// These nodes are mapped by `ItemLocalId` alongside the index of their parent node.
/// The HIR tree, including bodies, is pre-hashed.
pub struct OwnerNodes<'tcx> {
    /// Pre-computed hash of the full HIR. Used in the crate hash. Only present
    /// when incr. comp. is enabled.
    pub opt_hash_including_bodies: Option<Fingerprint>,
    /// Full HIR for the current owner.
    // The zeroth node's parent should never be accessed: the owner's parent is computed by the
    // hir_owner_parent query. It is set to `ItemLocalId::INVALID` to force an ICE if accidentally
    // used.
    pub nodes: IndexVec<ItemLocalId, ParentedNode<'tcx>>,
    /// Content of local bodies.
    pub bodies: SortedMap<ItemLocalId, &'tcx Body<'tcx>>,
}

impl<'tcx> OwnerNodes<'tcx> {
    pub fn node(&self) -> OwnerNode<'tcx> {
        // Indexing must ensure it is an OwnerNode.
        self.nodes[ItemLocalId::ZERO].node.as_owner().unwrap()
    }
}

impl fmt::Debug for OwnerNodes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnerNodes")
            // Do not print all the pointers to all the nodes, as it would be unreadable.
            .field("node", &self.nodes[ItemLocalId::ZERO])
            .field(
                "parents",
                &fmt::from_fn(|f| {
                    f.debug_list()
                        .entries(self.nodes.iter_enumerated().map(|(id, parented_node)| {
                            fmt::from_fn(move |f| write!(f, "({id:?}, {:?})", parented_node.parent))
                        }))
                        .finish()
                }),
            )
            .field("bodies", &self.bodies)
            .field("opt_hash_including_bodies", &self.opt_hash_including_bodies)
            .finish()
    }
}

/// Full information resulting from lowering an AST node.
#[derive(Debug, HashStable_Generic)]
pub struct OwnerInfo<'hir> {
    /// Contents of the HIR.
    pub nodes: OwnerNodes<'hir>,
    /// Map from each nested owner to its parent's local id.
    pub parenting: LocalDefIdMap<ItemLocalId>,
    /// Collected attributes of the HIR nodes.
    pub attrs: AttributeMap<'hir>,
    /// Map indicating what traits are in scope for places where this
    /// is relevant; generated by resolve.
    pub trait_map: ItemLocalMap<Box<[TraitCandidate]>>,
}

impl<'tcx> OwnerInfo<'tcx> {
    #[inline]
    pub fn node(&self) -> OwnerNode<'tcx> {
        self.nodes.node()
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum MaybeOwner<'tcx> {
    Owner(&'tcx OwnerInfo<'tcx>),
    NonOwner(HirId),
    /// Used as a placeholder for unused LocalDefId.
    Phantom,
}

impl<'tcx> MaybeOwner<'tcx> {
    pub fn as_owner(self) -> Option<&'tcx OwnerInfo<'tcx>> {
        match self {
            MaybeOwner::Owner(i) => Some(i),
            MaybeOwner::NonOwner(_) | MaybeOwner::Phantom => None,
        }
    }

    pub fn unwrap(self) -> &'tcx OwnerInfo<'tcx> {
        self.as_owner().unwrap_or_else(|| panic!("Not a HIR owner"))
    }
}

/// The top-level data structure that stores the entire contents of
/// the crate currently being compiled.
///
/// For more details, see the [rustc dev guide].
///
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/hir.html
#[derive(Debug)]
pub struct Crate<'hir> {
    pub owners: IndexVec<LocalDefId, MaybeOwner<'hir>>,
    // Only present when incr. comp. is enabled.
    pub opt_hir_hash: Option<Fingerprint>,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Closure<'hir> {
    pub def_id: LocalDefId,
    pub binder: ClosureBinder,
    pub constness: Constness,
    pub capture_clause: CaptureBy,
    pub bound_generic_params: &'hir [GenericParam<'hir>],
    pub fn_decl: &'hir FnDecl<'hir>,
    pub body: BodyId,
    /// The span of the declaration block: 'move |...| -> ...'
    pub fn_decl_span: Span,
    /// The span of the argument block `|...|`
    pub fn_arg_span: Option<Span>,
    pub kind: ClosureKind,
}

#[derive(Clone, PartialEq, Eq, Debug, Copy, Hash, HashStable_Generic, Encodable, Decodable)]
pub enum ClosureKind {
    /// This is a plain closure expression.
    Closure,
    /// This is a coroutine expression -- i.e. a closure expression in which
    /// we've found a `yield`. These can arise either from "plain" coroutine
    ///  usage (e.g. `let x = || { yield (); }`) or from a desugared expression
    /// (e.g. `async` and `gen` blocks).
    Coroutine(CoroutineKind),
    /// This is a coroutine-closure, which is a special sugared closure that
    /// returns one of the sugared coroutine (`async`/`gen`/`async gen`). It
    /// additionally allows capturing the coroutine's upvars by ref, and therefore
    /// needs to be specially treated during analysis and borrowck.
    CoroutineClosure(CoroutineDesugaring),
}

/// A block of statements `{ .. }`, which may have a label (in this case the
/// `targeted_by_break` field will be `true`) and may be `unsafe` by means of
/// the `rules` being anything but `DefaultBlock`.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Block<'hir> {
    /// Statements in a block.
    pub stmts: &'hir [Stmt<'hir>],
    /// An expression at the end of the block
    /// without a semicolon, if any.
    pub expr: Option<&'hir Expr<'hir>>,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// Distinguishes between `unsafe { ... }` and `{ ... }`.
    pub rules: BlockCheckMode,
    /// The span includes the curly braces `{` and `}` around the block.
    pub span: Span,
    /// If true, then there may exist `break 'a` values that aim to
    /// break out of this block early.
    /// Used by `'label: {}` blocks and by `try {}` blocks.
    pub targeted_by_break: bool,
}

impl<'hir> Block<'hir> {
    pub fn innermost_block(&self) -> &Block<'hir> {
        let mut block = self;
        while let Some(Expr { kind: ExprKind::Block(inner_block, _), .. }) = block.expr {
            block = inner_block;
        }
        block
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct TyPat<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: TyPatKind<'hir>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Pat<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: PatKind<'hir>,
    pub span: Span,
    /// Whether to use default binding modes.
    /// At present, this is false only for destructuring assignment.
    pub default_binding_modes: bool,
}

impl<'hir> Pat<'hir> {
    fn walk_short_(&self, it: &mut impl FnMut(&Pat<'hir>) -> bool) -> bool {
        if !it(self) {
            return false;
        }

        use PatKind::*;
        match self.kind {
            Missing => unreachable!(),
            Wild | Never | Expr(_) | Range(..) | Binding(.., None) | Err(_) => true,
            Box(s) | Deref(s) | Ref(s, _) | Binding(.., Some(s)) | Guard(s, _) => s.walk_short_(it),
            Struct(_, fields, _) => fields.iter().all(|field| field.pat.walk_short_(it)),
            TupleStruct(_, s, _) | Tuple(s, _) | Or(s) => s.iter().all(|p| p.walk_short_(it)),
            Slice(before, slice, after) => {
                before.iter().chain(slice).chain(after.iter()).all(|p| p.walk_short_(it))
            }
        }
    }

    /// Walk the pattern in left-to-right order,
    /// short circuiting (with `.all(..)`) if `false` is returned.
    ///
    /// Note that when visiting e.g. `Tuple(ps)`,
    /// if visiting `ps[0]` returns `false`,
    /// then `ps[1]` will not be visited.
    pub fn walk_short(&self, mut it: impl FnMut(&Pat<'hir>) -> bool) -> bool {
        self.walk_short_(&mut it)
    }

    fn walk_(&self, it: &mut impl FnMut(&Pat<'hir>) -> bool) {
        if !it(self) {
            return;
        }

        use PatKind::*;
        match self.kind {
            Missing | Wild | Never | Expr(_) | Range(..) | Binding(.., None) | Err(_) => {}
            Box(s) | Deref(s) | Ref(s, _) | Binding(.., Some(s)) | Guard(s, _) => s.walk_(it),
            Struct(_, fields, _) => fields.iter().for_each(|field| field.pat.walk_(it)),
            TupleStruct(_, s, _) | Tuple(s, _) | Or(s) => s.iter().for_each(|p| p.walk_(it)),
            Slice(before, slice, after) => {
                before.iter().chain(slice).chain(after.iter()).for_each(|p| p.walk_(it))
            }
        }
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If `it(pat)` returns `false`, the children are not visited.
    pub fn walk(&self, mut it: impl FnMut(&Pat<'hir>) -> bool) {
        self.walk_(&mut it)
    }

    /// Walk the pattern in left-to-right order.
    ///
    /// If you always want to recurse, prefer this method over `walk`.
    pub fn walk_always(&self, mut it: impl FnMut(&Pat<'_>)) {
        self.walk(|p| {
            it(p);
            true
        })
    }

    /// Whether this a never pattern.
    pub fn is_never_pattern(&self) -> bool {
        let mut is_never_pattern = false;
        self.walk(|pat| match &pat.kind {
            PatKind::Never => {
                is_never_pattern = true;
                false
            }
            PatKind::Or(s) => {
                is_never_pattern = s.iter().all(|p| p.is_never_pattern());
                false
            }
            _ => true,
        });
        is_never_pattern
    }
}

/// A single field in a struct pattern.
///
/// Patterns like the fields of Foo `{ x, ref y, ref mut z }`
/// are treated the same as` x: x, y: ref y, z: ref mut z`,
/// except `is_shorthand` is true.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct PatField<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    /// The identifier for the field.
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: &'hir Pat<'hir>,
    pub is_shorthand: bool,
    pub span: Span,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic, Hash, Eq, Encodable, Decodable)]
pub enum RangeEnd {
    Included,
    Excluded,
}

impl fmt::Display for RangeEnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RangeEnd::Included => "..=",
            RangeEnd::Excluded => "..",
        })
    }
}

// Equivalent to `Option<usize>`. That type takes up 16 bytes on 64-bit, but
// this type only takes up 4 bytes, at the cost of being restricted to a
// maximum value of `u32::MAX - 1`. In practice, this is more than enough.
#[derive(Clone, Copy, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct DotDotPos(u32);

impl DotDotPos {
    /// Panics if n >= u32::MAX.
    pub fn new(n: Option<usize>) -> Self {
        match n {
            Some(n) => {
                assert!(n < u32::MAX as usize);
                Self(n as u32)
            }
            None => Self(u32::MAX),
        }
    }

    pub fn as_opt_usize(&self) -> Option<usize> {
        if self.0 == u32::MAX { None } else { Some(self.0 as usize) }
    }
}

impl fmt::Debug for DotDotPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_opt_usize().fmt(f)
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct PatExpr<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    pub kind: PatExprKind<'hir>,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum PatExprKind<'hir> {
    Lit {
        lit: &'hir Lit,
        // FIXME: move this into `Lit` and handle negated literal expressions
        // once instead of matching on unop neg expressions everywhere.
        negated: bool,
    },
    ConstBlock(ConstBlock),
    /// A path pattern for a unit struct/variant or a (maybe-associated) constant.
    Path(QPath<'hir>),
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum TyPatKind<'hir> {
    /// A range pattern (e.g., `1..=2` or `1..2`).
    Range(&'hir ConstArg<'hir>, &'hir ConstArg<'hir>),

    /// A list of patterns where only one needs to be satisfied
    Or(&'hir [TyPat<'hir>]),

    /// A placeholder for a pattern that wasn't well formed in some way.
    Err(ErrorGuaranteed),
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum PatKind<'hir> {
    /// A missing pattern, e.g. for an anonymous param in a bare fn like `fn f(u32)`.
    Missing,

    /// Represents a wildcard pattern (i.e., `_`).
    Wild,

    /// A fresh binding `ref mut binding @ OPT_SUBPATTERN`.
    /// The `HirId` is the canonical ID for the variable being bound,
    /// (e.g., in `Ok(x) | Err(x)`, both `x` use the same canonical ID),
    /// which is the pattern ID of the first `x`.
    ///
    /// The `BindingMode` is what's provided by the user, before match
    /// ergonomics are applied. For the binding mode actually in use,
    /// see [`TypeckResults::extract_binding_mode`].
    ///
    /// [`TypeckResults::extract_binding_mode`]: ../../rustc_middle/ty/struct.TypeckResults.html#method.extract_binding_mode
    Binding(BindingMode, HirId, Ident, Option<&'hir Pat<'hir>>),

    /// A struct or struct variant pattern (e.g., `Variant {x, y, ..}`).
    /// The `bool` is `true` in the presence of a `..`.
    Struct(QPath<'hir>, &'hir [PatField<'hir>], bool),

    /// A tuple struct/variant pattern `Variant(x, y, .., z)`.
    /// If the `..` pattern fragment is present, then `DotDotPos` denotes its position.
    /// `0 <= position <= subpats.len()`
    TupleStruct(QPath<'hir>, &'hir [Pat<'hir>], DotDotPos),

    /// An or-pattern `A | B | C`.
    /// Invariant: `pats.len() >= 2`.
    Or(&'hir [Pat<'hir>]),

    /// A never pattern `!`.
    Never,

    /// A tuple pattern (e.g., `(a, b)`).
    /// If the `..` pattern fragment is present, then `DotDotPos` denotes its position.
    /// `0 <= position <= subpats.len()`
    Tuple(&'hir [Pat<'hir>], DotDotPos),

    /// A `box` pattern.
    Box(&'hir Pat<'hir>),

    /// A `deref` pattern (currently `deref!()` macro-based syntax).
    Deref(&'hir Pat<'hir>),

    /// A reference pattern (e.g., `&mut (a, b)`).
    Ref(&'hir Pat<'hir>, Mutability),

    /// A literal, const block or path.
    Expr(&'hir PatExpr<'hir>),

    /// A guard pattern (e.g., `x if guard(x)`).
    Guard(&'hir Pat<'hir>, &'hir Expr<'hir>),

    /// A range pattern (e.g., `1..=2` or `1..2`).
    Range(Option<&'hir PatExpr<'hir>>, Option<&'hir PatExpr<'hir>>, RangeEnd),

    /// A slice pattern, `[before_0, ..., before_n, (slice, after_0, ..., after_n)?]`.
    ///
    /// Here, `slice` is lowered from the syntax `($binding_mode $ident @)? ..`.
    /// If `slice` exists, then `after` can be non-empty.
    ///
    /// The representation for e.g., `[a, b, .., c, d]` is:
    /// ```ignore (illustrative)
    /// PatKind::Slice([Binding(a), Binding(b)], Some(Wild), [Binding(c), Binding(d)])
    /// ```
    Slice(&'hir [Pat<'hir>], Option<&'hir Pat<'hir>>, &'hir [Pat<'hir>]),

    /// A placeholder for a pattern that wasn't well formed in some way.
    Err(ErrorGuaranteed),
}

/// A statement.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Stmt<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: StmtKind<'hir>,
    pub span: Span,
}

/// The contents of a statement.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum StmtKind<'hir> {
    /// A local (`let`) binding.
    Let(&'hir LetStmt<'hir>),

    /// An item binding.
    Item(ItemId),

    /// An expression without a trailing semi-colon (must have unit type).
    Expr(&'hir Expr<'hir>),

    /// An expression with a trailing semi-colon (may have any type).
    Semi(&'hir Expr<'hir>),
}

/// Represents a `let` statement (i.e., `let <pat>:<ty> = <init>;`).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct LetStmt<'hir> {
    /// Span of `super` in `super let`.
    pub super_: Option<Span>,
    pub pat: &'hir Pat<'hir>,
    /// Type annotation, if any (otherwise the type will be inferred).
    pub ty: Option<&'hir Ty<'hir>>,
    /// Initializer expression to set the value, if any.
    pub init: Option<&'hir Expr<'hir>>,
    /// Else block for a `let...else` binding.
    pub els: Option<&'hir Block<'hir>>,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    /// Can be `ForLoopDesugar` if the `let` statement is part of a `for` loop
    /// desugaring, or `AssignDesugar` if it is the result of a complex
    /// assignment desugaring. Otherwise will be `Normal`.
    pub source: LocalSource,
}

/// Represents a single arm of a `match` expression, e.g.
/// `<pat> (if <guard>) => <body>`.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Arm<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    /// If this pattern and the optional guard matches, then `body` is evaluated.
    pub pat: &'hir Pat<'hir>,
    /// Optional guard clause.
    pub guard: Option<&'hir Expr<'hir>>,
    /// The expression the arm evaluates to if this arm matches.
    pub body: &'hir Expr<'hir>,
}

/// Represents a `let <pat>[: <ty>] = <expr>` expression (not a [`LetStmt`]), occurring in an `if-let`
/// or `let-else`, evaluating to a boolean. Typically the pattern is refutable.
///
/// In an `if let`, imagine it as `if (let <pat> = <expr>) { ... }`; in a let-else, it is part of
/// the desugaring to if-let. Only let-else supports the type annotation at present.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct LetExpr<'hir> {
    pub span: Span,
    pub pat: &'hir Pat<'hir>,
    pub ty: Option<&'hir Ty<'hir>>,
    pub init: &'hir Expr<'hir>,
    /// `Recovered::Yes` when this let expressions is not in a syntactically valid location.
    /// Used to prevent building MIR in such situations.
    pub recovered: ast::Recovered,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct ExprField<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub ident: Ident,
    pub expr: &'hir Expr<'hir>,
    pub span: Span,
    pub is_shorthand: bool,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic)]
pub enum BlockCheckMode {
    DefaultBlock,
    UnsafeBlock(UnsafeSource),
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic)]
pub enum UnsafeSource {
    CompilerGenerated,
    UserProvided,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct BodyId {
    pub hir_id: HirId,
}

/// The body of a function, closure, or constant value. In the case of
/// a function, the body contains not only the function body itself
/// (which is an expression), but also the argument patterns, since
/// those are something that the caller doesn't really care about.
///
/// # Examples
///
/// ```
/// fn foo((x, y): (u32, u32)) -> u32 {
///     x + y
/// }
/// ```
///
/// Here, the `Body` associated with `foo()` would contain:
///
/// - an `params` array containing the `(x, y)` pattern
/// - a `value` containing the `x + y` expression (maybe wrapped in a block)
/// - `coroutine_kind` would be `None`
///
/// All bodies have an **owner**, which can be accessed via the HIR
/// map using `body_owner_def_id()`.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Body<'hir> {
    pub params: &'hir [Param<'hir>],
    pub value: &'hir Expr<'hir>,
}

impl<'hir> Body<'hir> {
    pub fn id(&self) -> BodyId {
        BodyId { hir_id: self.value.hir_id }
    }
}

/// The type of source expression that caused this coroutine to be created.
#[derive(Clone, PartialEq, Eq, Debug, Copy, Hash, HashStable_Generic, Encodable, Decodable)]
pub enum CoroutineKind {
    /// A coroutine that comes from a desugaring.
    Desugared(CoroutineDesugaring, CoroutineSource),

    /// A coroutine literal created via a `yield` inside a closure.
    Coroutine(Movability),
}

impl CoroutineKind {
    pub fn movability(self) -> Movability {
        match self {
            CoroutineKind::Desugared(CoroutineDesugaring::Async, _)
            | CoroutineKind::Desugared(CoroutineDesugaring::AsyncGen, _) => Movability::Static,
            CoroutineKind::Desugared(CoroutineDesugaring::Gen, _) => Movability::Movable,
            CoroutineKind::Coroutine(mov) => mov,
        }
    }
}

impl CoroutineKind {
    pub fn is_fn_like(self) -> bool {
        matches!(self, CoroutineKind::Desugared(_, CoroutineSource::Fn))
    }
}

impl fmt::Display for CoroutineKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineKind::Desugared(d, k) => {
                d.fmt(f)?;
                k.fmt(f)
            }
            CoroutineKind::Coroutine(_) => f.write_str("coroutine"),
        }
    }
}

/// In the case of a coroutine created as part of an async/gen construct,
/// which kind of async/gen construct caused it to be created?
///
/// This helps error messages but is also used to drive coercions in
/// type-checking (see #60424).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Copy, HashStable_Generic, Encodable, Decodable)]
pub enum CoroutineSource {
    /// An explicit `async`/`gen` block written by the user.
    Block,

    /// An explicit `async`/`gen` closure written by the user.
    Closure,

    /// The `async`/`gen` block generated as the body of an async/gen function.
    Fn,
}

impl fmt::Display for CoroutineSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineSource::Block => "block",
            CoroutineSource::Closure => "closure body",
            CoroutineSource::Fn => "fn body",
        }
        .fmt(f)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Copy, Hash, HashStable_Generic, Encodable, Decodable)]
pub enum CoroutineDesugaring {
    /// An explicit `async` block or the body of an `async` function.
    Async,

    /// An explicit `gen` block or the body of a `gen` function.
    Gen,

    /// An explicit `async gen` block or the body of an `async gen` function,
    /// which is able to both `yield` and `.await`.
    AsyncGen,
}

impl fmt::Display for CoroutineDesugaring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoroutineDesugaring::Async => {
                if f.alternate() {
                    f.write_str("`async` ")?;
                } else {
                    f.write_str("async ")?
                }
            }
            CoroutineDesugaring::Gen => {
                if f.alternate() {
                    f.write_str("`gen` ")?;
                } else {
                    f.write_str("gen ")?
                }
            }
            CoroutineDesugaring::AsyncGen => {
                if f.alternate() {
                    f.write_str("`async gen` ")?;
                } else {
                    f.write_str("async gen ")?
                }
            }
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BodyOwnerKind {
    /// Functions and methods.
    Fn,

    /// Closures
    Closure,

    /// Constants and associated constants, also including inline constants.
    Const { inline: bool },

    /// Initializer of a `static` item.
    Static(Mutability),

    /// Fake body for a global asm to store its const-like value types.
    GlobalAsm,
}

impl BodyOwnerKind {
    pub fn is_fn_or_closure(self) -> bool {
        match self {
            BodyOwnerKind::Fn | BodyOwnerKind::Closure => true,
            BodyOwnerKind::Const { .. } | BodyOwnerKind::Static(_) | BodyOwnerKind::GlobalAsm => {
                false
            }
        }
    }
}

/// The kind of an item that requires const-checking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstContext {
    /// A `const fn`.
    ConstFn,

    /// A `static` or `static mut`.
    Static(Mutability),

    /// A `const`, associated `const`, or other const context.
    ///
    /// Other contexts include:
    /// - Array length expressions
    /// - Enum discriminants
    /// - Const generics
    ///
    /// For the most part, other contexts are treated just like a regular `const`, so they are
    /// lumped into the same category.
    Const { inline: bool },
}

impl ConstContext {
    /// A description of this const context that can appear between backticks in an error message.
    ///
    /// E.g. `const` or `static mut`.
    pub fn keyword_name(self) -> &'static str {
        match self {
            Self::Const { .. } => "const",
            Self::Static(Mutability::Not) => "static",
            Self::Static(Mutability::Mut) => "static mut",
            Self::ConstFn => "const fn",
        }
    }
}

/// A colloquial, trivially pluralizable description of this const context for use in error
/// messages.
impl fmt::Display for ConstContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Const { .. } => write!(f, "constant"),
            Self::Static(_) => write!(f, "static"),
            Self::ConstFn => write!(f, "constant function"),
        }
    }
}

// NOTE: `IntoDiagArg` impl for `ConstContext` lives in `rustc_errors`
// due to a cyclical dependency between hir and that crate.

/// A literal.
pub type Lit = Spanned<LitKind>;

/// A constant (expression) that's not an item or associated item,
/// but needs its own `DefId` for type-checking, const-eval, etc.
/// These are usually found nested inside types (e.g., array lengths)
/// or expressions (e.g., repeat counts), and also used to define
/// explicit discriminant values for enum variants.
///
/// You can check if this anon const is a default in a const param
/// `const N: usize = { ... }` with `tcx.hir_opt_const_param_default_param_def_id(..)`
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct AnonConst {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    pub body: BodyId,
    pub span: Span,
}

/// An inline constant expression `const { something }`.
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct ConstBlock {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    pub body: BodyId,
}

/// An expression.
///
/// For more details, see the [rust lang reference].
/// Note that the reference does not document nightly-only features.
/// There may be also slight differences in the names and representation of AST nodes between
/// the compiler and the reference.
///
/// [rust lang reference]: https://doc.rust-lang.org/reference/expressions.html
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Expr<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub kind: ExprKind<'hir>,
    pub span: Span,
}

impl Expr<'_> {
    pub fn precedence(&self) -> ExprPrecedence {
        match &self.kind {
            ExprKind::Closure(closure) => {
                match closure.fn_decl.output {
                    FnRetTy::DefaultReturn(_) => ExprPrecedence::Jump,
                    FnRetTy::Return(_) => ExprPrecedence::Unambiguous,
                }
            }

            ExprKind::Break(..)
            | ExprKind::Ret(..)
            | ExprKind::Yield(..)
            | ExprKind::Become(..) => ExprPrecedence::Jump,

            // Binop-like expr kinds, handled by `AssocOp`.
            ExprKind::Binary(op, ..) => op.node.precedence(),
            ExprKind::Cast(..) => ExprPrecedence::Cast,

            ExprKind::Assign(..) |
            ExprKind::AssignOp(..) => ExprPrecedence::Assign,

            // Unary, prefix
            ExprKind::AddrOf(..)
            // Here `let pats = expr` has `let pats =` as a "unary" prefix of `expr`.
            // However, this is not exactly right. When `let _ = a` is the LHS of a binop we
            // need parens sometimes. E.g. we can print `(let _ = a) && b` as `let _ = a && b`
            // but we need to print `(let _ = a) < b` as-is with parens.
            | ExprKind::Let(..)
            | ExprKind::Unary(..) => ExprPrecedence::Prefix,

            // Never need parens
            ExprKind::Array(_)
            | ExprKind::Block(..)
            | ExprKind::Call(..)
            | ExprKind::ConstBlock(_)
            | ExprKind::Continue(..)
            | ExprKind::Field(..)
            | ExprKind::If(..)
            | ExprKind::Index(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::Lit(_)
            | ExprKind::Loop(..)
            | ExprKind::Match(..)
            | ExprKind::MethodCall(..)
            | ExprKind::OffsetOf(..)
            | ExprKind::Path(..)
            | ExprKind::Repeat(..)
            | ExprKind::Struct(..)
            | ExprKind::Tup(_)
            | ExprKind::Type(..)
            | ExprKind::UnsafeBinderCast(..)
            | ExprKind::Use(..)
            | ExprKind::Err(_) => ExprPrecedence::Unambiguous,

            ExprKind::DropTemps(expr, ..) => expr.precedence(),
        }
    }

    /// Whether this looks like a place expr, without checking for deref
    /// adjustments.
    /// This will return `true` in some potentially surprising cases such as
    /// `CONSTANT.field`.
    pub fn is_syntactic_place_expr(&self) -> bool {
        self.is_place_expr(|_| true)
    }

    /// Whether this is a place expression.
    ///
    /// `allow_projections_from` should return `true` if indexing a field or index expression based
    /// on the given expression should be considered a place expression.
    pub fn is_place_expr(&self, mut allow_projections_from: impl FnMut(&Self) -> bool) -> bool {
        match self.kind {
            ExprKind::Path(QPath::Resolved(_, ref path)) => {
                matches!(path.res, Res::Local(..) | Res::Def(DefKind::Static { .. }, _) | Res::Err)
            }

            // Type ascription inherits its place expression kind from its
            // operand. See:
            // https://github.com/rust-lang/rfcs/blob/master/text/0803-type-ascription.md#type-ascription-and-temporaries
            ExprKind::Type(ref e, _) => e.is_place_expr(allow_projections_from),

            // Unsafe binder cast preserves place-ness of the sub-expression.
            ExprKind::UnsafeBinderCast(_, e, _) => e.is_place_expr(allow_projections_from),

            ExprKind::Unary(UnOp::Deref, _) => true,

            ExprKind::Field(ref base, _) | ExprKind::Index(ref base, _, _) => {
                allow_projections_from(base) || base.is_place_expr(allow_projections_from)
            }

            // Lang item paths cannot currently be local variables or statics.
            ExprKind::Path(QPath::LangItem(..)) => false,

            // Partially qualified paths in expressions can only legally
            // refer to associated items which are always rvalues.
            ExprKind::Path(QPath::TypeRelative(..))
            | ExprKind::Call(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Use(..)
            | ExprKind::Struct(..)
            | ExprKind::Tup(..)
            | ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::Closure { .. }
            | ExprKind::Block(..)
            | ExprKind::Repeat(..)
            | ExprKind::Array(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::Become(..)
            | ExprKind::Let(..)
            | ExprKind::Loop(..)
            | ExprKind::Assign(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::OffsetOf(..)
            | ExprKind::AssignOp(..)
            | ExprKind::Lit(_)
            | ExprKind::ConstBlock(..)
            | ExprKind::Unary(..)
            | ExprKind::AddrOf(..)
            | ExprKind::Binary(..)
            | ExprKind::Yield(..)
            | ExprKind::Cast(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err(_) => false,
        }
    }

    /// Check if expression is an integer literal that can be used
    /// where `usize` is expected.
    pub fn is_size_lit(&self) -> bool {
        matches!(
            self.kind,
            ExprKind::Lit(Lit {
                node: LitKind::Int(_, LitIntType::Unsuffixed | LitIntType::Unsigned(UintTy::Usize)),
                ..
            })
        )
    }

    /// If `Self.kind` is `ExprKind::DropTemps(expr)`, drill down until we get a non-`DropTemps`
    /// `Expr`. This is used in suggestions to ignore this `ExprKind` as it is semantically
    /// silent, only signaling the ownership system. By doing this, suggestions that check the
    /// `ExprKind` of any given `Expr` for presentation don't have to care about `DropTemps`
    /// beyond remembering to call this function before doing analysis on it.
    pub fn peel_drop_temps(&self) -> &Self {
        let mut expr = self;
        while let ExprKind::DropTemps(inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn peel_blocks(&self) -> &Self {
        let mut expr = self;
        while let ExprKind::Block(Block { expr: Some(inner), .. }, _) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn peel_borrows(&self) -> &Self {
        let mut expr = self;
        while let ExprKind::AddrOf(.., inner) = &expr.kind {
            expr = inner;
        }
        expr
    }

    pub fn can_have_side_effects(&self) -> bool {
        match self.peel_drop_temps().kind {
            ExprKind::Path(_) | ExprKind::Lit(_) | ExprKind::OffsetOf(..) | ExprKind::Use(..) => {
                false
            }
            ExprKind::Type(base, _)
            | ExprKind::Unary(_, base)
            | ExprKind::Field(base, _)
            | ExprKind::Index(base, _, _)
            | ExprKind::AddrOf(.., base)
            | ExprKind::Cast(base, _)
            | ExprKind::UnsafeBinderCast(_, base, _) => {
                // This isn't exactly true for `Index` and all `Unary`, but we are using this
                // method exclusively for diagnostics and there's a *cultural* pressure against
                // them being used only for its side-effects.
                base.can_have_side_effects()
            }
            ExprKind::Struct(_, fields, init) => {
                let init_side_effects = match init {
                    StructTailExpr::Base(init) => init.can_have_side_effects(),
                    StructTailExpr::DefaultFields(_) | StructTailExpr::None => false,
                };
                fields.iter().map(|field| field.expr).any(|e| e.can_have_side_effects())
                    || init_side_effects
            }

            ExprKind::Array(args)
            | ExprKind::Tup(args)
            | ExprKind::Call(
                Expr {
                    kind:
                        ExprKind::Path(QPath::Resolved(
                            None,
                            Path { res: Res::Def(DefKind::Ctor(_, CtorKind::Fn), _), .. },
                        )),
                    ..
                },
                args,
            ) => args.iter().any(|arg| arg.can_have_side_effects()),
            ExprKind::If(..)
            | ExprKind::Match(..)
            | ExprKind::MethodCall(..)
            | ExprKind::Call(..)
            | ExprKind::Closure { .. }
            | ExprKind::Block(..)
            | ExprKind::Repeat(..)
            | ExprKind::Break(..)
            | ExprKind::Continue(..)
            | ExprKind::Ret(..)
            | ExprKind::Become(..)
            | ExprKind::Let(..)
            | ExprKind::Loop(..)
            | ExprKind::Assign(..)
            | ExprKind::InlineAsm(..)
            | ExprKind::AssignOp(..)
            | ExprKind::ConstBlock(..)
            | ExprKind::Binary(..)
            | ExprKind::Yield(..)
            | ExprKind::DropTemps(..)
            | ExprKind::Err(_) => true,
        }
    }

    /// To a first-order approximation, is this a pattern?
    pub fn is_approximately_pattern(&self) -> bool {
        match &self.kind {
            ExprKind::Array(_)
            | ExprKind::Call(..)
            | ExprKind::Tup(_)
            | ExprKind::Lit(_)
            | ExprKind::Path(_)
            | ExprKind::Struct(..) => true,
            _ => false,
        }
    }

    /// Whether this and the `other` expression are the same for purposes of an indexing operation.
    ///
    /// This is only used for diagnostics to see if we have things like `foo[i]` where `foo` is
    /// borrowed multiple times with `i`.
    pub fn equivalent_for_indexing(&self, other: &Expr<'_>) -> bool {
        match (self.kind, other.kind) {
            (ExprKind::Lit(lit1), ExprKind::Lit(lit2)) => lit1.node == lit2.node,
            (
                ExprKind::Path(QPath::LangItem(item1, _)),
                ExprKind::Path(QPath::LangItem(item2, _)),
            ) => item1 == item2,
            (
                ExprKind::Path(QPath::Resolved(None, path1)),
                ExprKind::Path(QPath::Resolved(None, path2)),
            ) => path1.res == path2.res,
            (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeTo, _),
                    [val1],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeTo, _),
                    [val2],
                    StructTailExpr::None,
                ),
            )
            | (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeToInclusive, _),
                    [val1],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeToInclusive, _),
                    [val2],
                    StructTailExpr::None,
                ),
            )
            | (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeFrom, _),
                    [val1],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeFrom, _),
                    [val2],
                    StructTailExpr::None,
                ),
            )
            | (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeFromCopy, _),
                    [val1],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeFromCopy, _),
                    [val2],
                    StructTailExpr::None,
                ),
            ) => val1.expr.equivalent_for_indexing(val2.expr),
            (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::Range, _),
                    [val1, val3],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::Range, _),
                    [val2, val4],
                    StructTailExpr::None,
                ),
            )
            | (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeCopy, _),
                    [val1, val3],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeCopy, _),
                    [val2, val4],
                    StructTailExpr::None,
                ),
            )
            | (
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeInclusiveCopy, _),
                    [val1, val3],
                    StructTailExpr::None,
                ),
                ExprKind::Struct(
                    QPath::LangItem(LangItem::RangeInclusiveCopy, _),
                    [val2, val4],
                    StructTailExpr::None,
                ),
            ) => {
                val1.expr.equivalent_for_indexing(val2.expr)
                    && val3.expr.equivalent_for_indexing(val4.expr)
            }
            _ => false,
        }
    }

    pub fn method_ident(&self) -> Option<Ident> {
        match self.kind {
            ExprKind::MethodCall(receiver_method, ..) => Some(receiver_method.ident),
            ExprKind::Unary(_, expr) | ExprKind::AddrOf(.., expr) => expr.method_ident(),
            _ => None,
        }
    }
}

/// Checks if the specified expression is a built-in range literal.
/// (See: `LoweringContext::lower_expr()`).
pub fn is_range_literal(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // All built-in range literals but `..=` and `..` desugar to `Struct`s.
        ExprKind::Struct(ref qpath, _, _) => matches!(
            **qpath,
            QPath::LangItem(
                LangItem::Range
                    | LangItem::RangeTo
                    | LangItem::RangeFrom
                    | LangItem::RangeFull
                    | LangItem::RangeToInclusive
                    | LangItem::RangeCopy
                    | LangItem::RangeFromCopy
                    | LangItem::RangeInclusiveCopy,
                ..
            )
        ),

        // `..=` desugars into `::std::ops::RangeInclusive::new(...)`.
        ExprKind::Call(ref func, _) => {
            matches!(func.kind, ExprKind::Path(QPath::LangItem(LangItem::RangeInclusiveNew, ..)))
        }

        _ => false,
    }
}

/// Checks if the specified expression needs parentheses for prefix
/// or postfix suggestions to be valid.
/// For example, `a + b` requires parentheses to suggest `&(a + b)`,
/// but just `a` does not.
/// Similarly, `(a + b).c()` also requires parentheses.
/// This should not be used for other types of suggestions.
pub fn expr_needs_parens(expr: &Expr<'_>) -> bool {
    match expr.kind {
        // parenthesize if needed (Issue #46756)
        ExprKind::Cast(_, _) | ExprKind::Binary(_, _, _) => true,
        // parenthesize borrows of range literals (Issue #54505)
        _ if is_range_literal(expr) => true,
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum ExprKind<'hir> {
    /// Allow anonymous constants from an inline `const` block
    ConstBlock(ConstBlock),
    /// An array (e.g., `[a, b, c, d]`).
    Array(&'hir [Expr<'hir>]),
    /// A function call.
    ///
    /// The first field resolves to the function itself (usually an `ExprKind::Path`),
    /// and the second field is the list of arguments.
    /// This also represents calling the constructor of
    /// tuple-like ADTs such as tuple structs and enum variants.
    Call(&'hir Expr<'hir>, &'hir [Expr<'hir>]),
    /// A method call (e.g., `x.foo::<'static, Bar, Baz>(a, b, c, d)`).
    ///
    /// The `PathSegment` represents the method name and its generic arguments
    /// (within the angle brackets).
    /// The `&Expr` is the expression that evaluates
    /// to the object on which the method is being called on (the receiver),
    /// and the `&[Expr]` is the rest of the arguments.
    /// Thus, `x.foo::<Bar, Baz>(a, b, c, d)` is represented as
    /// `ExprKind::MethodCall(PathSegment { foo, [Bar, Baz] }, x, [a, b, c, d], span)`.
    /// The final `Span` represents the span of the function and arguments
    /// (e.g. `foo::<Bar, Baz>(a, b, c, d)` in `x.foo::<Bar, Baz>(a, b, c, d)`
    ///
    /// To resolve the called method to a `DefId`, call [`type_dependent_def_id`] with
    /// the `hir_id` of the `MethodCall` node itself.
    ///
    /// [`type_dependent_def_id`]: ../../rustc_middle/ty/struct.TypeckResults.html#method.type_dependent_def_id
    MethodCall(&'hir PathSegment<'hir>, &'hir Expr<'hir>, &'hir [Expr<'hir>], Span),
    /// An use expression (e.g., `var.use`).
    Use(&'hir Expr<'hir>, Span),
    /// A tuple (e.g., `(a, b, c, d)`).
    Tup(&'hir [Expr<'hir>]),
    /// A binary operation (e.g., `a + b`, `a * b`).
    Binary(BinOp, &'hir Expr<'hir>, &'hir Expr<'hir>),
    /// A unary operation (e.g., `!x`, `*x`).
    Unary(UnOp, &'hir Expr<'hir>),
    /// A literal (e.g., `1`, `"foo"`).
    Lit(&'hir Lit),
    /// A cast (e.g., `foo as f64`).
    Cast(&'hir Expr<'hir>, &'hir Ty<'hir>),
    /// A type ascription (e.g., `x: Foo`). See RFC 3307.
    Type(&'hir Expr<'hir>, &'hir Ty<'hir>),
    /// Wraps the expression in a terminating scope.
    /// This makes it semantically equivalent to `{ let _t = expr; _t }`.
    ///
    /// This construct only exists to tweak the drop order in AST lowering.
    /// An example of that is the desugaring of `for` loops.
    DropTemps(&'hir Expr<'hir>),
    /// A `let $pat = $expr` expression.
    ///
    /// These are not [`LetStmt`] and only occur as expressions.
    /// The `let Some(x) = foo()` in `if let Some(x) = foo()` is an example of `Let(..)`.
    Let(&'hir LetExpr<'hir>),
    /// An `if` block, with an optional else block.
    ///
    /// I.e., `if <expr> { <expr> } else { <expr> }`.
    ///
    /// The "then" expr is always `ExprKind::Block`. If present, the "else" expr is always
    /// `ExprKind::Block` (for `else`) or `ExprKind::If` (for `else if`).
    /// Note that using an `Expr` instead of a `Block` for the "then" part is intentional,
    /// as it simplifies the type coercion machinery.
    If(&'hir Expr<'hir>, &'hir Expr<'hir>, Option<&'hir Expr<'hir>>),
    /// A conditionless loop (can be exited with `break`, `continue`, or `return`).
    ///
    /// I.e., `'label: loop { <block> }`.
    ///
    /// The `Span` is the loop header (`for x in y`/`while let pat = expr`).
    Loop(&'hir Block<'hir>, Option<Label>, LoopSource, Span),
    /// A `match` block, with a source that indicates whether or not it is
    /// the result of a desugaring, and if so, which kind.
    Match(&'hir Expr<'hir>, &'hir [Arm<'hir>], MatchSource),
    /// A closure (e.g., `move |a, b, c| {a + b + c}`).
    ///
    /// The `Span` is the argument block `|...|`.
    ///
    /// This may also be a coroutine literal or an `async block` as indicated by the
    /// `Option<Movability>`.
    Closure(&'hir Closure<'hir>),
    /// A block (e.g., `'label: { ... }`).
    Block(&'hir Block<'hir>, Option<Label>),

    /// An assignment (e.g., `a = foo()`).
    Assign(&'hir Expr<'hir>, &'hir Expr<'hir>, Span),
    /// An assignment with an operator.
    ///
    /// E.g., `a += 1`.
    AssignOp(AssignOp, &'hir Expr<'hir>, &'hir Expr<'hir>),
    /// Access of a named (e.g., `obj.foo`) or unnamed (e.g., `obj.0`) struct or tuple field.
    Field(&'hir Expr<'hir>, Ident),
    /// An indexing operation (`foo[2]`).
    /// Similar to [`ExprKind::MethodCall`], the final `Span` represents the span of the brackets
    /// and index.
    Index(&'hir Expr<'hir>, &'hir Expr<'hir>, Span),

    /// Path to a definition, possibly containing lifetime or type parameters.
    Path(QPath<'hir>),

    /// A referencing operation (i.e., `&a` or `&mut a`).
    AddrOf(BorrowKind, Mutability, &'hir Expr<'hir>),
    /// A `break`, with an optional label to break.
    Break(Destination, Option<&'hir Expr<'hir>>),
    /// A `continue`, with an optional label.
    Continue(Destination),
    /// A `return`, with an optional value to be returned.
    Ret(Option<&'hir Expr<'hir>>),
    /// A `become`, with the value to be returned.
    Become(&'hir Expr<'hir>),

    /// Inline assembly (from `asm!`), with its outputs and inputs.
    InlineAsm(&'hir InlineAsm<'hir>),

    /// Field offset (`offset_of!`)
    OffsetOf(&'hir Ty<'hir>, &'hir [Ident]),

    /// A struct or struct-like variant literal expression.
    ///
    /// E.g., `Foo {x: 1, y: 2}`, or `Foo {x: 1, .. base}`,
    /// where `base` is the `Option<Expr>`.
    Struct(&'hir QPath<'hir>, &'hir [ExprField<'hir>], StructTailExpr<'hir>),

    /// An array literal constructed from one repeated element.
    ///
    /// E.g., `[1; 5]`. The first expression is the element
    /// to be repeated; the second is the number of times to repeat it.
    Repeat(&'hir Expr<'hir>, &'hir ConstArg<'hir>),

    /// A suspension point for coroutines (i.e., `yield <expr>`).
    Yield(&'hir Expr<'hir>, YieldSource),

    /// Operators which can be used to interconvert `unsafe` binder types.
    /// e.g. `unsafe<'a> &'a i32` <=> `&i32`.
    UnsafeBinderCast(UnsafeBinderCastKind, &'hir Expr<'hir>, Option<&'hir Ty<'hir>>),

    /// A placeholder for an expression that wasn't syntactically well formed in some way.
    Err(rustc_span::ErrorGuaranteed),
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum StructTailExpr<'hir> {
    /// A struct expression where all the fields are explicitly enumerated: `Foo { a, b }`.
    None,
    /// A struct expression with a "base", an expression of the same type as the outer struct that
    /// will be used to populate any fields not explicitly mentioned: `Foo { ..base }`
    Base(&'hir Expr<'hir>),
    /// A struct expression with a `..` tail but no "base" expression. The values from the struct
    /// fields' default values will be used to populate any fields not explicitly mentioned:
    /// `Foo { .. }`.
    DefaultFields(Span),
}

/// Represents an optionally `Self`-qualified value/type path or associated extension.
///
/// To resolve the path to a `DefId`, call [`qpath_res`].
///
/// [`qpath_res`]: ../../rustc_middle/ty/struct.TypeckResults.html#method.qpath_res
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum QPath<'hir> {
    /// Path to a definition, optionally "fully-qualified" with a `Self`
    /// type, if the path points to an associated item in a trait.
    ///
    /// E.g., an unqualified path like `Clone::clone` has `None` for `Self`,
    /// while `<Vec<T> as Clone>::clone` has `Some(Vec<T>)` for `Self`,
    /// even though they both have the same two-segment `Clone::clone` `Path`.
    Resolved(Option<&'hir Ty<'hir>>, &'hir Path<'hir>),

    /// Type-related paths (e.g., `<T>::default` or `<T>::Output`).
    /// Will be resolved by type-checking to an associated item.
    ///
    /// UFCS source paths can desugar into this, with `Vec::new` turning into
    /// `<Vec>::new`, and `T::X::Y::method` into `<<<T>::X>::Y>::method`,
    /// the `X` and `Y` nodes each being a `TyKind::Path(QPath::TypeRelative(..))`.
    TypeRelative(&'hir Ty<'hir>, &'hir PathSegment<'hir>),

    /// Reference to a `#[lang = "foo"]` item.
    LangItem(LangItem, Span),
}

impl<'hir> QPath<'hir> {
    /// Returns the span of this `QPath`.
    pub fn span(&self) -> Span {
        match *self {
            QPath::Resolved(_, path) => path.span,
            QPath::TypeRelative(qself, ps) => qself.span.to(ps.ident.span),
            QPath::LangItem(_, span) => span,
        }
    }

    /// Returns the span of the qself of this `QPath`. For example, `()` in
    /// `<() as Trait>::method`.
    pub fn qself_span(&self) -> Span {
        match *self {
            QPath::Resolved(_, path) => path.span,
            QPath::TypeRelative(qself, _) => qself.span,
            QPath::LangItem(_, span) => span,
        }
    }
}

/// Hints at the original code for a let statement.
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum LocalSource {
    /// A `match _ { .. }`.
    Normal,
    /// When lowering async functions, we create locals within the `async move` so that
    /// all parameters are dropped after the future is polled.
    ///
    /// ```ignore (pseudo-Rust)
    /// async fn foo(<pattern> @ x: Type) {
    ///     async move {
    ///         let <pattern> = x;
    ///     }
    /// }
    /// ```
    AsyncFn,
    /// A desugared `<expr>.await`.
    AwaitDesugar,
    /// A desugared `expr = expr`, where the LHS is a tuple, struct, array or underscore expression.
    /// The span is that of the `=` sign.
    AssignDesugar(Span),
    /// A contract `#[ensures(..)]` attribute injects a let binding for the check that runs at point of return.
    Contract,
}

/// Hints at the original code for a `match _ { .. }`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic, Encodable, Decodable)]
pub enum MatchSource {
    /// A `match _ { .. }`.
    Normal,
    /// A `expr.match { .. }`.
    Postfix,
    /// A desugared `for _ in _ { .. }` loop.
    ForLoopDesugar,
    /// A desugared `?` operator.
    TryDesugar(HirId),
    /// A desugared `<expr>.await`.
    AwaitDesugar,
    /// A desugared `format_args!()`.
    FormatArgs,
}

impl MatchSource {
    #[inline]
    pub const fn name(self) -> &'static str {
        use MatchSource::*;
        match self {
            Normal => "match",
            Postfix => ".match",
            ForLoopDesugar => "for",
            TryDesugar(_) => "?",
            AwaitDesugar => ".await",
            FormatArgs => "format_args!()",
        }
    }
}

/// The loop type that yielded an `ExprKind::Loop`.
#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic)]
pub enum LoopSource {
    /// A `loop { .. }` loop.
    Loop,
    /// A `while _ { .. }` loop.
    While,
    /// A `for _ in _ { .. }` loop.
    ForLoop,
}

impl LoopSource {
    pub fn name(self) -> &'static str {
        match self {
            LoopSource::Loop => "loop",
            LoopSource::While => "while",
            LoopSource::ForLoop => "for",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, HashStable_Generic)]
pub enum LoopIdError {
    OutsideLoopScope,
    UnlabeledCfInWhileCondition,
    UnresolvedLabel,
}

impl fmt::Display for LoopIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LoopIdError::OutsideLoopScope => "not inside loop scope",
            LoopIdError::UnlabeledCfInWhileCondition => {
                "unlabeled control flow (break or continue) in while condition"
            }
            LoopIdError::UnresolvedLabel => "label not found",
        })
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct Destination {
    /// This is `Some(_)` iff there is an explicit user-specified 'label
    pub label: Option<Label>,

    /// These errors are caught and then reported during the diagnostics pass in
    /// `librustc_passes/loops.rs`
    pub target_id: Result<HirId, LoopIdError>,
}

/// The yield kind that caused an `ExprKind::Yield`.
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum YieldSource {
    /// An `<expr>.await`.
    Await { expr: Option<HirId> },
    /// A plain `yield`.
    Yield,
}

impl fmt::Display for YieldSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            YieldSource::Await { .. } => "`await`",
            YieldSource::Yield => "`yield`",
        })
    }
}

// N.B., if you change this, you'll probably want to change the corresponding
// type structure in middle/ty.rs as well.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct MutTy<'hir> {
    pub ty: &'hir Ty<'hir>,
    pub mutbl: Mutability,
}

/// Represents a function's signature in a trait declaration,
/// trait implementation, or a free function.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct FnSig<'hir> {
    pub header: FnHeader,
    pub decl: &'hir FnDecl<'hir>,
    pub span: Span,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct TraitItemId {
    pub owner_id: OwnerId,
}

impl TraitItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }
}

/// Represents an item declaration within a trait declaration,
/// possibly including a default implementation. A trait item is
/// either required (meaning it doesn't have an implementation, just a
/// signature) or provided (meaning it has a default implementation).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct TraitItem<'hir> {
    pub ident: Ident,
    pub owner_id: OwnerId,
    pub generics: &'hir Generics<'hir>,
    pub kind: TraitItemKind<'hir>,
    pub span: Span,
    pub defaultness: Defaultness,
}

macro_rules! expect_methods_self_kind {
    ( $( $name:ident, $ret_ty:ty, $pat:pat, $ret_val:expr; )* ) => {
        $(
            #[track_caller]
            pub fn $name(&self) -> $ret_ty {
                let $pat = &self.kind else { expect_failed(stringify!($ident), self) };
                $ret_val
            }
        )*
    }
}

macro_rules! expect_methods_self {
    ( $( $name:ident, $ret_ty:ty, $pat:pat, $ret_val:expr; )* ) => {
        $(
            #[track_caller]
            pub fn $name(&self) -> $ret_ty {
                let $pat = self else { expect_failed(stringify!($ident), self) };
                $ret_val
            }
        )*
    }
}

#[track_caller]
fn expect_failed<T: fmt::Debug>(ident: &'static str, found: T) -> ! {
    panic!("{ident}: found {found:?}")
}

impl<'hir> TraitItem<'hir> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }

    pub fn trait_item_id(&self) -> TraitItemId {
        TraitItemId { owner_id: self.owner_id }
    }

    expect_methods_self_kind! {
        expect_const, (&'hir Ty<'hir>, Option<BodyId>),
            TraitItemKind::Const(ty, body), (ty, *body);

        expect_fn, (&FnSig<'hir>, &TraitFn<'hir>),
            TraitItemKind::Fn(ty, trfn), (ty, trfn);

        expect_type, (GenericBounds<'hir>, Option<&'hir Ty<'hir>>),
            TraitItemKind::Type(bounds, ty), (bounds, *ty);
    }
}

/// Represents a trait method's body (or just argument names).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum TraitFn<'hir> {
    /// No default body in the trait, just a signature.
    Required(&'hir [Option<Ident>]),

    /// Both signature and body are provided in the trait.
    Provided(BodyId),
}

/// Represents a trait method or associated constant or type
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum TraitItemKind<'hir> {
    /// An associated constant with an optional value (otherwise `impl`s must contain a value).
    Const(&'hir Ty<'hir>, Option<BodyId>),
    /// An associated function with an optional body.
    Fn(FnSig<'hir>, TraitFn<'hir>),
    /// An associated type with (possibly empty) bounds and optional concrete
    /// type.
    Type(GenericBounds<'hir>, Option<&'hir Ty<'hir>>),
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct ImplItemId {
    pub owner_id: OwnerId,
}

impl ImplItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }
}

/// Represents an associated item within an impl block.
///
/// Refer to [`Impl`] for an impl block declaration.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct ImplItem<'hir> {
    pub ident: Ident,
    pub owner_id: OwnerId,
    pub generics: &'hir Generics<'hir>,
    pub kind: ImplItemKind<'hir>,
    pub defaultness: Defaultness,
    pub span: Span,
    pub vis_span: Span,
}

impl<'hir> ImplItem<'hir> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }

    pub fn impl_item_id(&self) -> ImplItemId {
        ImplItemId { owner_id: self.owner_id }
    }

    expect_methods_self_kind! {
        expect_const, (&'hir Ty<'hir>, BodyId), ImplItemKind::Const(ty, body), (ty, *body);
        expect_fn,    (&FnSig<'hir>, BodyId),   ImplItemKind::Fn(ty, body),    (ty, *body);
        expect_type,  &'hir Ty<'hir>,           ImplItemKind::Type(ty),        ty;
    }
}

/// Represents various kinds of content within an `impl`.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum ImplItemKind<'hir> {
    /// An associated constant of the given type, set to the constant result
    /// of the expression.
    Const(&'hir Ty<'hir>, BodyId),
    /// An associated function implementation with the given signature and body.
    Fn(FnSig<'hir>, BodyId),
    /// An associated type.
    Type(&'hir Ty<'hir>),
}

/// A constraint on an associated item.
///
/// ### Examples
///
/// * the `A = Ty` and `B = Ty` in `Trait<A = Ty, B = Ty>`
/// * the `G<Ty> = Ty` in `Trait<G<Ty> = Ty>`
/// * the `A: Bound` in `Trait<A: Bound>`
/// * the `RetTy` in `Trait(ArgTy, ArgTy) -> RetTy`
/// * the `C = { Ct }` in `Trait<C = { Ct }>` (feature `associated_const_equality`)
/// * the `f(..): Bound` in `Trait<f(..): Bound>` (feature `return_type_notation`)
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct AssocItemConstraint<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub ident: Ident,
    pub gen_args: &'hir GenericArgs<'hir>,
    pub kind: AssocItemConstraintKind<'hir>,
    pub span: Span,
}

impl<'hir> AssocItemConstraint<'hir> {
    /// Obtain the type on the RHS of an assoc ty equality constraint if applicable.
    pub fn ty(self) -> Option<&'hir Ty<'hir>> {
        match self.kind {
            AssocItemConstraintKind::Equality { term: Term::Ty(ty) } => Some(ty),
            _ => None,
        }
    }

    /// Obtain the const on the RHS of an assoc const equality constraint if applicable.
    pub fn ct(self) -> Option<&'hir ConstArg<'hir>> {
        match self.kind {
            AssocItemConstraintKind::Equality { term: Term::Const(ct) } => Some(ct),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum Term<'hir> {
    Ty(&'hir Ty<'hir>),
    Const(&'hir ConstArg<'hir>),
}

impl<'hir> From<&'hir Ty<'hir>> for Term<'hir> {
    fn from(ty: &'hir Ty<'hir>) -> Self {
        Term::Ty(ty)
    }
}

impl<'hir> From<&'hir ConstArg<'hir>> for Term<'hir> {
    fn from(c: &'hir ConstArg<'hir>) -> Self {
        Term::Const(c)
    }
}

/// The kind of [associated item constraint][AssocItemConstraint].
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum AssocItemConstraintKind<'hir> {
    /// An equality constraint for an associated item (e.g., `AssocTy = Ty` in `Trait<AssocTy = Ty>`).
    ///
    /// Also known as an *associated item binding* (we *bind* an associated item to a term).
    ///
    /// Furthermore, associated type equality constraints can also be referred to as *associated type
    /// bindings*. Similarly with associated const equality constraints and *associated const bindings*.
    Equality { term: Term<'hir> },
    /// A bound on an associated type (e.g., `AssocTy: Bound` in `Trait<AssocTy: Bound>`).
    Bound { bounds: &'hir [GenericBound<'hir>] },
}

impl<'hir> AssocItemConstraintKind<'hir> {
    pub fn descr(&self) -> &'static str {
        match self {
            AssocItemConstraintKind::Equality { .. } => "binding",
            AssocItemConstraintKind::Bound { .. } => "constraint",
        }
    }
}

/// An uninhabited enum used to make `Infer` variants on [`Ty`] and [`ConstArg`] be
/// unreachable. Zero-Variant enums are guaranteed to have the same layout as the never
/// type.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum AmbigArg {}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
#[repr(C)]
/// Represents a type in the `HIR`.
///
/// The `Unambig` generic parameter represents whether the position this type is from is
/// unambiguously a type or ambiguous as to whether it is a type or a const. When in an
/// ambiguous context the parameter is instantiated with an uninhabited type making the
/// [`TyKind::Infer`] variant unusable and [`GenericArg::Infer`] is used instead.
pub struct Ty<'hir, Unambig = ()> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub span: Span,
    pub kind: TyKind<'hir, Unambig>,
}

impl<'hir> Ty<'hir, AmbigArg> {
    /// Converts a `Ty` in an ambiguous position to one in an unambiguous position.
    ///
    /// Functions accepting an unambiguous types may expect the [`TyKind::Infer`] variant
    /// to be used. Care should be taken to separately handle infer types when calling this
    /// function as it cannot be handled by downstream code making use of the returned ty.
    ///
    /// In practice this may mean overriding the [`Visitor::visit_infer`][visit_infer] method on hir visitors, or
    /// specifically matching on [`GenericArg::Infer`] when handling generic arguments.
    ///
    /// [visit_infer]: [rustc_hir::intravisit::Visitor::visit_infer]
    pub fn as_unambig_ty(&self) -> &Ty<'hir> {
        // SAFETY: `Ty` is `repr(C)` and `TyKind` is marked `repr(u8)` so that the layout is
        // the same across different ZST type arguments.
        let ptr = self as *const Ty<'hir, AmbigArg> as *const Ty<'hir, ()>;
        unsafe { &*ptr }
    }
}

impl<'hir> Ty<'hir> {
    /// Converts a `Ty` in an unambigous position to one in an ambiguous position. This is
    /// fallible as the [`TyKind::Infer`] variant is not present in ambiguous positions.
    ///
    /// Functions accepting ambiguous types will not handle the [`TyKind::Infer`] variant, if
    /// infer types are relevant to you then care should be taken to handle them separately.
    pub fn try_as_ambig_ty(&self) -> Option<&Ty<'hir, AmbigArg>> {
        if let TyKind::Infer(()) = self.kind {
            return None;
        }

        // SAFETY: `Ty` is `repr(C)` and `TyKind` is marked `repr(u8)` so that the layout is
        // the same across different ZST type arguments. We also asserted that the `self` is
        // not a `TyKind::Infer` so there is no risk of transmuting a `()` to `AmbigArg`.
        let ptr = self as *const Ty<'hir> as *const Ty<'hir, AmbigArg>;
        Some(unsafe { &*ptr })
    }
}

impl<'hir> Ty<'hir, AmbigArg> {
    pub fn peel_refs(&self) -> &Ty<'hir> {
        let mut final_ty = self.as_unambig_ty();
        while let TyKind::Ref(_, MutTy { ty, .. }) = &final_ty.kind {
            final_ty = ty;
        }
        final_ty
    }
}

impl<'hir> Ty<'hir> {
    pub fn peel_refs(&self) -> &Self {
        let mut final_ty = self;
        while let TyKind::Ref(_, MutTy { ty, .. }) = &final_ty.kind {
            final_ty = ty;
        }
        final_ty
    }

    /// Returns `true` if `param_def_id` matches the `bounded_ty` of this predicate.
    pub fn as_generic_param(&self) -> Option<(DefId, Ident)> {
        let TyKind::Path(QPath::Resolved(None, path)) = self.kind else {
            return None;
        };
        let [segment] = &path.segments else {
            return None;
        };
        match path.res {
            Res::Def(DefKind::TyParam, def_id) | Res::SelfTyParam { trait_: def_id } => {
                Some((def_id, segment.ident))
            }
            _ => None,
        }
    }

    pub fn find_self_aliases(&self) -> Vec<Span> {
        use crate::intravisit::Visitor;
        struct MyVisitor(Vec<Span>);
        impl<'v> Visitor<'v> for MyVisitor {
            fn visit_ty(&mut self, t: &'v Ty<'v, AmbigArg>) {
                if matches!(
                    &t.kind,
                    TyKind::Path(QPath::Resolved(
                        _,
                        Path { res: crate::def::Res::SelfTyAlias { .. }, .. },
                    ))
                ) {
                    self.0.push(t.span);
                    return;
                }
                crate::intravisit::walk_ty(self, t);
            }
        }

        let mut my_visitor = MyVisitor(vec![]);
        my_visitor.visit_ty_unambig(self);
        my_visitor.0
    }

    /// Whether `ty` is a type with `_` placeholders that can be inferred. Used in diagnostics only to
    /// use inference to provide suggestions for the appropriate type if possible.
    pub fn is_suggestable_infer_ty(&self) -> bool {
        fn are_suggestable_generic_args(generic_args: &[GenericArg<'_>]) -> bool {
            generic_args.iter().any(|arg| match arg {
                GenericArg::Type(ty) => ty.as_unambig_ty().is_suggestable_infer_ty(),
                GenericArg::Infer(_) => true,
                _ => false,
            })
        }
        debug!(?self);
        match &self.kind {
            TyKind::Infer(()) => true,
            TyKind::Slice(ty) => ty.is_suggestable_infer_ty(),
            TyKind::Array(ty, length) => {
                ty.is_suggestable_infer_ty() || matches!(length.kind, ConstArgKind::Infer(..))
            }
            TyKind::Tup(tys) => tys.iter().any(Self::is_suggestable_infer_ty),
            TyKind::Ptr(mut_ty) | TyKind::Ref(_, mut_ty) => mut_ty.ty.is_suggestable_infer_ty(),
            TyKind::Path(QPath::TypeRelative(ty, segment)) => {
                ty.is_suggestable_infer_ty() || are_suggestable_generic_args(segment.args().args)
            }
            TyKind::Path(QPath::Resolved(ty_opt, Path { segments, .. })) => {
                ty_opt.is_some_and(Self::is_suggestable_infer_ty)
                    || segments
                        .iter()
                        .any(|segment| are_suggestable_generic_args(segment.args().args))
            }
            _ => false,
        }
    }
}

/// Not represented directly in the AST; referred to by name through a `ty_path`.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Hash, Debug, HashStable_Generic)]
pub enum PrimTy {
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Str,
    Bool,
    Char,
}

impl PrimTy {
    /// All of the primitive types
    pub const ALL: [Self; 19] = [
        // any changes here should also be reflected in `PrimTy::from_name`
        Self::Int(IntTy::I8),
        Self::Int(IntTy::I16),
        Self::Int(IntTy::I32),
        Self::Int(IntTy::I64),
        Self::Int(IntTy::I128),
        Self::Int(IntTy::Isize),
        Self::Uint(UintTy::U8),
        Self::Uint(UintTy::U16),
        Self::Uint(UintTy::U32),
        Self::Uint(UintTy::U64),
        Self::Uint(UintTy::U128),
        Self::Uint(UintTy::Usize),
        Self::Float(FloatTy::F16),
        Self::Float(FloatTy::F32),
        Self::Float(FloatTy::F64),
        Self::Float(FloatTy::F128),
        Self::Bool,
        Self::Char,
        Self::Str,
    ];

    /// Like [`PrimTy::name`], but returns a &str instead of a symbol.
    ///
    /// Used by clippy.
    pub fn name_str(self) -> &'static str {
        match self {
            PrimTy::Int(i) => i.name_str(),
            PrimTy::Uint(u) => u.name_str(),
            PrimTy::Float(f) => f.name_str(),
            PrimTy::Str => "str",
            PrimTy::Bool => "bool",
            PrimTy::Char => "char",
        }
    }

    pub fn name(self) -> Symbol {
        match self {
            PrimTy::Int(i) => i.name(),
            PrimTy::Uint(u) => u.name(),
            PrimTy::Float(f) => f.name(),
            PrimTy::Str => sym::str,
            PrimTy::Bool => sym::bool,
            PrimTy::Char => sym::char,
        }
    }

    /// Returns the matching `PrimTy` for a `Symbol` such as "str" or "i32".
    /// Returns `None` if no matching type is found.
    pub fn from_name(name: Symbol) -> Option<Self> {
        let ty = match name {
            // any changes here should also be reflected in `PrimTy::ALL`
            sym::i8 => Self::Int(IntTy::I8),
            sym::i16 => Self::Int(IntTy::I16),
            sym::i32 => Self::Int(IntTy::I32),
            sym::i64 => Self::Int(IntTy::I64),
            sym::i128 => Self::Int(IntTy::I128),
            sym::isize => Self::Int(IntTy::Isize),
            sym::u8 => Self::Uint(UintTy::U8),
            sym::u16 => Self::Uint(UintTy::U16),
            sym::u32 => Self::Uint(UintTy::U32),
            sym::u64 => Self::Uint(UintTy::U64),
            sym::u128 => Self::Uint(UintTy::U128),
            sym::usize => Self::Uint(UintTy::Usize),
            sym::f16 => Self::Float(FloatTy::F16),
            sym::f32 => Self::Float(FloatTy::F32),
            sym::f64 => Self::Float(FloatTy::F64),
            sym::f128 => Self::Float(FloatTy::F128),
            sym::bool => Self::Bool,
            sym::char => Self::Char,
            sym::str => Self::Str,
            _ => return None,
        };
        Some(ty)
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct BareFnTy<'hir> {
    pub safety: Safety,
    pub abi: ExternAbi,
    pub generic_params: &'hir [GenericParam<'hir>],
    pub decl: &'hir FnDecl<'hir>,
    // `Option` because bare fn parameter identifiers are optional. We also end up
    // with `None` in some error cases, e.g. invalid parameter patterns.
    pub param_idents: &'hir [Option<Ident>],
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct UnsafeBinderTy<'hir> {
    pub generic_params: &'hir [GenericParam<'hir>],
    pub inner_ty: &'hir Ty<'hir>,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct OpaqueTy<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    pub bounds: GenericBounds<'hir>,
    pub origin: OpaqueTyOrigin<LocalDefId>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, HashStable_Generic, Encodable, Decodable)]
pub enum PreciseCapturingArgKind<T, U> {
    Lifetime(T),
    /// Non-lifetime argument (type or const)
    Param(U),
}

pub type PreciseCapturingArg<'hir> =
    PreciseCapturingArgKind<&'hir Lifetime, PreciseCapturingNonLifetimeArg>;

impl PreciseCapturingArg<'_> {
    pub fn hir_id(self) -> HirId {
        match self {
            PreciseCapturingArg::Lifetime(lt) => lt.hir_id,
            PreciseCapturingArg::Param(param) => param.hir_id,
        }
    }

    pub fn name(self) -> Symbol {
        match self {
            PreciseCapturingArg::Lifetime(lt) => lt.ident.name,
            PreciseCapturingArg::Param(param) => param.ident.name,
        }
    }
}

/// We need to have a [`Node`] for the [`HirId`] that we attach the type/const param
/// resolution to. Lifetimes don't have this problem, and for them, it's actually
/// kind of detrimental to use a custom node type versus just using [`Lifetime`],
/// since resolve_bound_vars operates on `Lifetime`s.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct PreciseCapturingNonLifetimeArg {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub ident: Ident,
    pub res: Res,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum RpitContext {
    Trait,
    TraitImpl,
}

/// From whence the opaque type came.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum OpaqueTyOrigin<D> {
    /// `-> impl Trait`
    FnReturn {
        /// The defining function.
        parent: D,
        // Whether this is an RPITIT (return position impl trait in trait)
        in_trait_or_impl: Option<RpitContext>,
    },
    /// `async fn`
    AsyncFn {
        /// The defining function.
        parent: D,
        // Whether this is an AFIT (async fn in trait)
        in_trait_or_impl: Option<RpitContext>,
    },
    /// type aliases: `type Foo = impl Trait;`
    TyAlias {
        /// The type alias or associated type parent of the TAIT/ATPIT
        parent: D,
        /// associated types in impl blocks for traits.
        in_assoc_ty: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, HashStable_Generic)]
pub enum InferDelegationKind {
    Input(usize),
    Output,
}

/// The various kinds of types recognized by the compiler.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
// SAFETY: `repr(u8)` is required so that `TyKind<()>` and `TyKind<!>` are layout compatible
#[repr(u8, C)]
pub enum TyKind<'hir, Unambig = ()> {
    /// Actual type should be inherited from `DefId` signature
    InferDelegation(DefId, InferDelegationKind),
    /// A variable length slice (i.e., `[T]`).
    Slice(&'hir Ty<'hir>),
    /// A fixed length array (i.e., `[T; n]`).
    Array(&'hir Ty<'hir>, &'hir ConstArg<'hir>),
    /// A raw pointer (i.e., `*const T` or `*mut T`).
    Ptr(MutTy<'hir>),
    /// A reference (i.e., `&'a T` or `&'a mut T`).
    Ref(&'hir Lifetime, MutTy<'hir>),
    /// A bare function (e.g., `fn(usize) -> bool`).
    BareFn(&'hir BareFnTy<'hir>),
    /// An unsafe binder type (e.g. `unsafe<'a> Foo<'a>`).
    UnsafeBinder(&'hir UnsafeBinderTy<'hir>),
    /// The never type (`!`).
    Never,
    /// A tuple (`(A, B, C, D, ...)`).
    Tup(&'hir [Ty<'hir>]),
    /// A path to a type definition (`module::module::...::Type`), or an
    /// associated type (e.g., `<Vec<T> as Trait>::Type` or `<T>::Target`).
    ///
    /// Type parameters may be stored in each `PathSegment`.
    Path(QPath<'hir>),
    /// An opaque type definition itself. This is only used for `impl Trait`.
    OpaqueDef(&'hir OpaqueTy<'hir>),
    /// A trait ascription type, which is `impl Trait` within a local binding.
    TraitAscription(GenericBounds<'hir>),
    /// A trait object type `Bound1 + Bound2 + Bound3`
    /// where `Bound` is a trait or a lifetime.
    ///
    /// We use pointer tagging to represent a `&'hir Lifetime` and `TraitObjectSyntax` pair
    /// as otherwise this type being `repr(C)` would result in `TyKind` increasing in size.
    TraitObject(&'hir [PolyTraitRef<'hir>], TaggedRef<'hir, Lifetime, TraitObjectSyntax>),
    /// Unused for now.
    Typeof(&'hir AnonConst),
    /// Placeholder for a type that has failed to be defined.
    Err(rustc_span::ErrorGuaranteed),
    /// Pattern types (`pattern_type!(u32 is 1..)`)
    Pat(&'hir Ty<'hir>, &'hir TyPat<'hir>),
    /// `TyKind::Infer` means the type should be inferred instead of it having been
    /// specified. This can appear anywhere in a type.
    ///
    /// This variant is not always used to represent inference types, sometimes
    /// [`GenericArg::Infer`] is used instead.
    Infer(Unambig),
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum InlineAsmOperand<'hir> {
    In {
        reg: InlineAsmRegOrRegClass,
        expr: &'hir Expr<'hir>,
    },
    Out {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: Option<&'hir Expr<'hir>>,
    },
    InOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        expr: &'hir Expr<'hir>,
    },
    SplitInOut {
        reg: InlineAsmRegOrRegClass,
        late: bool,
        in_expr: &'hir Expr<'hir>,
        out_expr: Option<&'hir Expr<'hir>>,
    },
    Const {
        anon_const: ConstBlock,
    },
    SymFn {
        expr: &'hir Expr<'hir>,
    },
    SymStatic {
        path: QPath<'hir>,
        def_id: DefId,
    },
    Label {
        block: &'hir Block<'hir>,
    },
}

impl<'hir> InlineAsmOperand<'hir> {
    pub fn reg(&self) -> Option<InlineAsmRegOrRegClass> {
        match *self {
            Self::In { reg, .. }
            | Self::Out { reg, .. }
            | Self::InOut { reg, .. }
            | Self::SplitInOut { reg, .. } => Some(reg),
            Self::Const { .. }
            | Self::SymFn { .. }
            | Self::SymStatic { .. }
            | Self::Label { .. } => None,
        }
    }

    pub fn is_clobber(&self) -> bool {
        matches!(
            self,
            InlineAsmOperand::Out { reg: InlineAsmRegOrRegClass::Reg(_), late: _, expr: None }
        )
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct InlineAsm<'hir> {
    pub asm_macro: ast::AsmMacro,
    pub template: &'hir [InlineAsmTemplatePiece],
    pub template_strs: &'hir [(Symbol, Option<Symbol>, Span)],
    pub operands: &'hir [(InlineAsmOperand<'hir>, Span)],
    pub options: InlineAsmOptions,
    pub line_spans: &'hir [Span],
}

impl InlineAsm<'_> {
    pub fn contains_label(&self) -> bool {
        self.operands.iter().any(|x| matches!(x.0, InlineAsmOperand::Label { .. }))
    }
}

/// Represents a parameter in a function header.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Param<'hir> {
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub pat: &'hir Pat<'hir>,
    pub ty_span: Span,
    pub span: Span,
}

/// Represents the header (not the body) of a function declaration.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct FnDecl<'hir> {
    /// The types of the function's parameters.
    ///
    /// Additional argument data is stored in the function's [body](Body::params).
    pub inputs: &'hir [Ty<'hir>],
    pub output: FnRetTy<'hir>,
    pub c_variadic: bool,
    /// Does the function have an implicit self?
    pub implicit_self: ImplicitSelfKind,
    /// Is lifetime elision allowed.
    pub lifetime_elision_allowed: bool,
}

impl<'hir> FnDecl<'hir> {
    pub fn opt_delegation_sig_id(&self) -> Option<DefId> {
        if let FnRetTy::Return(ty) = self.output
            && let TyKind::InferDelegation(sig_id, _) = ty.kind
        {
            return Some(sig_id);
        }
        None
    }
}

/// Represents what type of implicit self a function has, if any.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum ImplicitSelfKind {
    /// Represents a `fn x(self);`.
    Imm,
    /// Represents a `fn x(mut self);`.
    Mut,
    /// Represents a `fn x(&self);`.
    RefImm,
    /// Represents a `fn x(&mut self);`.
    RefMut,
    /// Represents when a function does not have a self argument or
    /// when a function has a `self: X` argument.
    None,
}

impl ImplicitSelfKind {
    /// Does this represent an implicit self?
    pub fn has_implicit_self(&self) -> bool {
        !matches!(*self, ImplicitSelfKind::None)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub enum IsAsync {
    Async(Span),
    NotAsync,
}

impl IsAsync {
    pub fn is_async(self) -> bool {
        matches!(self, IsAsync::Async(_))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Defaultness {
    Default { has_value: bool },
    Final,
}

impl Defaultness {
    pub fn has_value(&self) -> bool {
        match *self {
            Defaultness::Default { has_value } => has_value,
            Defaultness::Final => true,
        }
    }

    pub fn is_final(&self) -> bool {
        *self == Defaultness::Final
    }

    pub fn is_default(&self) -> bool {
        matches!(*self, Defaultness::Default { .. })
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum FnRetTy<'hir> {
    /// Return type is not specified.
    ///
    /// Functions default to `()` and
    /// closures default to inference. Span points to where return
    /// type would be inserted.
    DefaultReturn(Span),
    /// Everything else.
    Return(&'hir Ty<'hir>),
}

impl<'hir> FnRetTy<'hir> {
    #[inline]
    pub fn span(&self) -> Span {
        match *self {
            Self::DefaultReturn(span) => span,
            Self::Return(ref ty) => ty.span,
        }
    }

    pub fn is_suggestable_infer_ty(&self) -> Option<&'hir Ty<'hir>> {
        if let Self::Return(ty) = self
            && ty.is_suggestable_infer_ty()
        {
            return Some(*ty);
        }
        None
    }
}

/// Represents `for<...>` binder before a closure
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum ClosureBinder {
    /// Binder is not specified.
    Default,
    /// Binder is specified.
    ///
    /// Span points to the whole `for<...>`.
    For { span: Span },
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Mod<'hir> {
    pub spans: ModSpans,
    pub item_ids: &'hir [ItemId],
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct ModSpans {
    /// A span from the first token past `{` to the last token until `}`.
    /// For `mod foo;`, the inner span ranges from the first token
    /// to the last token in the external file.
    pub inner_span: Span,
    pub inject_use_span: Span,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct EnumDef<'hir> {
    pub variants: &'hir [Variant<'hir>],
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Variant<'hir> {
    /// Name of the variant.
    pub ident: Ident,
    /// Id of the variant (not the constructor, see `VariantData::ctor_hir_id()`).
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    /// Fields and constructor id of the variant.
    pub data: VariantData<'hir>,
    /// Explicit discriminant (e.g., `Foo = 1`).
    pub disr_expr: Option<&'hir AnonConst>,
    /// Span
    pub span: Span,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic)]
pub enum UseKind {
    /// One import, e.g., `use foo::bar` or `use foo::bar as baz`.
    /// Also produced for each element of a list `use`, e.g.
    /// `use foo::{a, b}` lowers to `use foo::a; use foo::b;`.
    ///
    /// The identifier is the name defined by the import. E.g. for `use
    /// foo::bar` it is `bar`, for `use foo::bar as baz` it is `baz`.
    Single(Ident),

    /// Glob import, e.g., `use foo::*`.
    Glob,

    /// Degenerate list import, e.g., `use foo::{a, b}` produces
    /// an additional `use foo::{}` for performing checks such as
    /// unstable feature gating. May be removed in the future.
    ListStem,
}

/// References to traits in impls.
///
/// `resolve` maps each `TraitRef`'s `ref_id` to its defining trait; that's all
/// that the `ref_id` is for. Note that `ref_id`'s value is not the `HirId` of the
/// trait being referred to but just a unique `HirId` that serves as a key
/// within the resolution map.
#[derive(Clone, Debug, Copy, HashStable_Generic)]
pub struct TraitRef<'hir> {
    pub path: &'hir Path<'hir>,
    // Don't hash the `ref_id`. It is tracked via the thing it is used to access.
    #[stable_hasher(ignore)]
    pub hir_ref_id: HirId,
}

impl TraitRef<'_> {
    /// Gets the `DefId` of the referenced trait. It _must_ actually be a trait or trait alias.
    pub fn trait_def_id(&self) -> Option<DefId> {
        match self.path.res {
            Res::Def(DefKind::Trait | DefKind::TraitAlias, did) => Some(did),
            Res::Err => None,
            res => panic!("{res:?} did not resolve to a trait or trait alias"),
        }
    }
}

#[derive(Clone, Debug, Copy, HashStable_Generic)]
pub struct PolyTraitRef<'hir> {
    /// The `'a` in `for<'a> Foo<&'a T>`.
    pub bound_generic_params: &'hir [GenericParam<'hir>],

    /// The constness and polarity of the trait ref.
    ///
    /// The `async` modifier is lowered directly into a different trait for now.
    pub modifiers: TraitBoundModifiers,

    /// The `Foo<&'a T>` in `for<'a> Foo<&'a T>`.
    pub trait_ref: TraitRef<'hir>,

    pub span: Span,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct FieldDef<'hir> {
    pub span: Span,
    pub vis_span: Span,
    pub ident: Ident,
    #[stable_hasher(ignore)]
    pub hir_id: HirId,
    pub def_id: LocalDefId,
    pub ty: &'hir Ty<'hir>,
    pub safety: Safety,
    pub default: Option<&'hir AnonConst>,
}

impl FieldDef<'_> {
    // Still necessary in couple of places
    pub fn is_positional(&self) -> bool {
        self.ident.as_str().as_bytes()[0].is_ascii_digit()
    }
}

/// Fields and constructor IDs of enum variants and structs.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum VariantData<'hir> {
    /// A struct variant.
    ///
    /// E.g., `Bar { .. }` as in `enum Foo { Bar { .. } }`.
    Struct { fields: &'hir [FieldDef<'hir>], recovered: ast::Recovered },
    /// A tuple variant.
    ///
    /// E.g., `Bar(..)` as in `enum Foo { Bar(..) }`.
    Tuple(&'hir [FieldDef<'hir>], #[stable_hasher(ignore)] HirId, LocalDefId),
    /// A unit variant.
    ///
    /// E.g., `Bar = ..` as in `enum Foo { Bar = .. }`.
    Unit(#[stable_hasher(ignore)] HirId, LocalDefId),
}

impl<'hir> VariantData<'hir> {
    /// Return the fields of this variant.
    pub fn fields(&self) -> &'hir [FieldDef<'hir>] {
        match *self {
            VariantData::Struct { fields, .. } | VariantData::Tuple(fields, ..) => fields,
            _ => &[],
        }
    }

    pub fn ctor(&self) -> Option<(CtorKind, HirId, LocalDefId)> {
        match *self {
            VariantData::Tuple(_, hir_id, def_id) => Some((CtorKind::Fn, hir_id, def_id)),
            VariantData::Unit(hir_id, def_id) => Some((CtorKind::Const, hir_id, def_id)),
            VariantData::Struct { .. } => None,
        }
    }

    #[inline]
    pub fn ctor_kind(&self) -> Option<CtorKind> {
        self.ctor().map(|(kind, ..)| kind)
    }

    /// Return the `HirId` of this variant's constructor, if it has one.
    #[inline]
    pub fn ctor_hir_id(&self) -> Option<HirId> {
        self.ctor().map(|(_, hir_id, _)| hir_id)
    }

    /// Return the `LocalDefId` of this variant's constructor, if it has one.
    #[inline]
    pub fn ctor_def_id(&self) -> Option<LocalDefId> {
        self.ctor().map(|(.., def_id)| def_id)
    }
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, Hash, HashStable_Generic)]
pub struct ItemId {
    pub owner_id: OwnerId,
}

impl ItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }
}

/// An item
///
/// For more details, see the [rust lang reference].
/// Note that the reference does not document nightly-only features.
/// There may be also slight differences in the names and representation of AST nodes between
/// the compiler and the reference.
///
/// [rust lang reference]: https://doc.rust-lang.org/reference/items.html
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Item<'hir> {
    pub owner_id: OwnerId,
    pub kind: ItemKind<'hir>,
    pub span: Span,
    pub vis_span: Span,
}

impl<'hir> Item<'hir> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }

    pub fn item_id(&self) -> ItemId {
        ItemId { owner_id: self.owner_id }
    }

    /// Check if this is an [`ItemKind::Enum`], [`ItemKind::Struct`] or
    /// [`ItemKind::Union`].
    pub fn is_adt(&self) -> bool {
        matches!(self.kind, ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..))
    }

    /// Check if this is an [`ItemKind::Struct`] or [`ItemKind::Union`].
    pub fn is_struct_or_union(&self) -> bool {
        matches!(self.kind, ItemKind::Struct(..) | ItemKind::Union(..))
    }

    expect_methods_self_kind! {
        expect_extern_crate, (Option<Symbol>, Ident),
            ItemKind::ExternCrate(s, ident), (*s, *ident);

        expect_use, (&'hir UsePath<'hir>, UseKind), ItemKind::Use(p, uk), (p, *uk);

        expect_static, (Ident, &'hir Ty<'hir>, Mutability, BodyId),
            ItemKind::Static(ident, ty, mutbl, body), (*ident, ty, *mutbl, *body);

        expect_const, (Ident, &'hir Ty<'hir>, &'hir Generics<'hir>, BodyId),
            ItemKind::Const(ident, ty, generics, body), (*ident, ty, generics, *body);

        expect_fn, (Ident, &FnSig<'hir>, &'hir Generics<'hir>, BodyId),
            ItemKind::Fn { ident, sig, generics, body, .. }, (*ident, sig, generics, *body);

        expect_macro, (Ident, &ast::MacroDef, MacroKind),
            ItemKind::Macro(ident, def, mk), (*ident, def, *mk);

        expect_mod, (Ident, &'hir Mod<'hir>), ItemKind::Mod(ident, m), (*ident, m);

        expect_foreign_mod, (ExternAbi, &'hir [ForeignItemRef]),
            ItemKind::ForeignMod { abi, items }, (*abi, items);

        expect_global_asm, &'hir InlineAsm<'hir>, ItemKind::GlobalAsm { asm, .. }, asm;

        expect_ty_alias, (Ident, &'hir Ty<'hir>, &'hir Generics<'hir>),
            ItemKind::TyAlias(ident, ty, generics), (*ident, ty, generics);

        expect_enum, (Ident, &EnumDef<'hir>, &'hir Generics<'hir>),
            ItemKind::Enum(ident, def, generics), (*ident, def, generics);

        expect_struct, (Ident, &VariantData<'hir>, &'hir Generics<'hir>),
            ItemKind::Struct(ident, data, generics), (*ident, data, generics);

        expect_union, (Ident, &VariantData<'hir>, &'hir Generics<'hir>),
            ItemKind::Union(ident, data, generics), (*ident, data, generics);

        expect_trait,
            (
                IsAuto,
                Safety,
                Ident,
                &'hir Generics<'hir>,
                GenericBounds<'hir>,
                &'hir [TraitItemRef]
            ),
            ItemKind::Trait(is_auto, safety, ident, generics, bounds, items),
            (*is_auto, *safety, *ident, generics, bounds, items);

        expect_trait_alias, (Ident, &'hir Generics<'hir>, GenericBounds<'hir>),
            ItemKind::TraitAlias(ident, generics, bounds), (*ident, generics, bounds);

        expect_impl, &'hir Impl<'hir>, ItemKind::Impl(imp), imp;
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum Safety {
    Unsafe,
    Safe,
}

impl Safety {
    pub fn prefix_str(self) -> &'static str {
        match self {
            Self::Unsafe => "unsafe ",
            Self::Safe => "",
        }
    }

    #[inline]
    pub fn is_unsafe(self) -> bool {
        !self.is_safe()
    }

    #[inline]
    pub fn is_safe(self) -> bool {
        match self {
            Self::Unsafe => false,
            Self::Safe => true,
        }
    }
}

impl fmt::Display for Safety {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Self::Unsafe => "unsafe",
            Self::Safe => "safe",
        })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Encodable, Decodable, HashStable_Generic)]
pub enum Constness {
    Const,
    NotConst,
}

impl fmt::Display for Constness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Self::Const => "const",
            Self::NotConst => "non-const",
        })
    }
}

/// The actualy safety specified in syntax. We may treat
/// its safety different within the type system to create a
/// "sound by default" system that needs checking this enum
/// explicitly to allow unsafe operations.
#[derive(Copy, Clone, Debug, HashStable_Generic, PartialEq, Eq)]
pub enum HeaderSafety {
    /// A safe function annotated with `#[target_features]`.
    /// The type system treats this function as an unsafe function,
    /// but safety checking will check this enum to treat it as safe
    /// and allowing calling other safe target feature functions with
    /// the same features without requiring an additional unsafe block.
    SafeTargetFeatures,
    Normal(Safety),
}

impl From<Safety> for HeaderSafety {
    fn from(v: Safety) -> Self {
        Self::Normal(v)
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub struct FnHeader {
    pub safety: HeaderSafety,
    pub constness: Constness,
    pub asyncness: IsAsync,
    pub abi: ExternAbi,
}

impl FnHeader {
    pub fn is_async(&self) -> bool {
        matches!(self.asyncness, IsAsync::Async(_))
    }

    pub fn is_const(&self) -> bool {
        matches!(self.constness, Constness::Const)
    }

    pub fn is_unsafe(&self) -> bool {
        self.safety().is_unsafe()
    }

    pub fn is_safe(&self) -> bool {
        self.safety().is_safe()
    }

    pub fn safety(&self) -> Safety {
        match self.safety {
            HeaderSafety::SafeTargetFeatures => Safety::Unsafe,
            HeaderSafety::Normal(safety) => safety,
        }
    }
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum ItemKind<'hir> {
    /// An `extern crate` item, with optional *original* crate name if the crate was renamed.
    ///
    /// E.g., `extern crate foo` or `extern crate foo_bar as foo`.
    ExternCrate(Option<Symbol>, Ident),

    /// `use foo::bar::*;` or `use foo::bar::baz as quux;`
    ///
    /// or just
    ///
    /// `use foo::bar::baz;` (with `as baz` implicitly on the right).
    Use(&'hir UsePath<'hir>, UseKind),

    /// A `static` item.
    Static(Ident, &'hir Ty<'hir>, Mutability, BodyId),
    /// A `const` item.
    Const(Ident, &'hir Ty<'hir>, &'hir Generics<'hir>, BodyId),
    /// A function declaration.
    Fn {
        ident: Ident,
        sig: FnSig<'hir>,
        generics: &'hir Generics<'hir>,
        body: BodyId,
        /// Whether this function actually has a body.
        /// For functions without a body, `body` is synthesized (to avoid ICEs all over the
        /// compiler), but that code should never be translated.
        has_body: bool,
    },
    /// A MBE macro definition (`macro_rules!` or `macro`).
    Macro(Ident, &'hir ast::MacroDef, MacroKind),
    /// A module.
    Mod(Ident, &'hir Mod<'hir>),
    /// An external module, e.g. `extern { .. }`.
    ForeignMod { abi: ExternAbi, items: &'hir [ForeignItemRef] },
    /// Module-level inline assembly (from `global_asm!`).
    GlobalAsm {
        asm: &'hir InlineAsm<'hir>,
        /// A fake body which stores typeck results for the global asm's sym_fn
        /// operands, which are represented as path expressions. This body contains
        /// a single [`ExprKind::InlineAsm`] which points to the asm in the field
        /// above, and which is typechecked like a inline asm expr just for the
        /// typeck results.
        fake_body: BodyId,
    },
    /// A type alias, e.g., `type Foo = Bar<u8>`.
    TyAlias(Ident, &'hir Ty<'hir>, &'hir Generics<'hir>),
    /// An enum definition, e.g., `enum Foo<A, B> { C<A>, D<B> }`.
    Enum(Ident, EnumDef<'hir>, &'hir Generics<'hir>),
    /// A struct definition, e.g., `struct Foo<A> {x: A}`.
    Struct(Ident, VariantData<'hir>, &'hir Generics<'hir>),
    /// A union definition, e.g., `union Foo<A, B> {x: A, y: B}`.
    Union(Ident, VariantData<'hir>, &'hir Generics<'hir>),
    /// A trait definition.
    Trait(IsAuto, Safety, Ident, &'hir Generics<'hir>, GenericBounds<'hir>, &'hir [TraitItemRef]),
    /// A trait alias.
    TraitAlias(Ident, &'hir Generics<'hir>, GenericBounds<'hir>),

    /// An implementation, e.g., `impl<A> Trait for Foo { .. }`.
    Impl(&'hir Impl<'hir>),
}

/// Represents an impl block declaration.
///
/// E.g., `impl $Type { .. }` or `impl $Trait for $Type { .. }`
/// Refer to [`ImplItem`] for an associated item within an impl block.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct Impl<'hir> {
    pub constness: Constness,
    pub safety: Safety,
    pub polarity: ImplPolarity,
    pub defaultness: Defaultness,
    // We do not put a `Span` in `Defaultness` because it breaks foreign crate metadata
    // decoding as `Span`s cannot be decoded when a `Session` is not available.
    pub defaultness_span: Option<Span>,
    pub generics: &'hir Generics<'hir>,

    /// The trait being implemented, if any.
    pub of_trait: Option<TraitRef<'hir>>,

    pub self_ty: &'hir Ty<'hir>,
    pub items: &'hir [ImplItemRef],
}

impl ItemKind<'_> {
    pub fn ident(&self) -> Option<Ident> {
        match *self {
            ItemKind::ExternCrate(_, ident)
            | ItemKind::Use(_, UseKind::Single(ident))
            | ItemKind::Static(ident, ..)
            | ItemKind::Const(ident, ..)
            | ItemKind::Fn { ident, .. }
            | ItemKind::Macro(ident, ..)
            | ItemKind::Mod(ident, ..)
            | ItemKind::TyAlias(ident, ..)
            | ItemKind::Enum(ident, ..)
            | ItemKind::Struct(ident, ..)
            | ItemKind::Union(ident, ..)
            | ItemKind::Trait(_, _, ident, ..)
            | ItemKind::TraitAlias(ident, ..) => Some(ident),

            ItemKind::Use(_, UseKind::Glob | UseKind::ListStem)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm { .. }
            | ItemKind::Impl(_) => None,
        }
    }

    pub fn generics(&self) -> Option<&Generics<'_>> {
        Some(match self {
            ItemKind::Fn { generics, .. }
            | ItemKind::TyAlias(_, _, generics)
            | ItemKind::Const(_, _, generics, _)
            | ItemKind::Enum(_, _, generics)
            | ItemKind::Struct(_, _, generics)
            | ItemKind::Union(_, _, generics)
            | ItemKind::Trait(_, _, _, generics, _, _)
            | ItemKind::TraitAlias(_, generics, _)
            | ItemKind::Impl(Impl { generics, .. }) => generics,
            _ => return None,
        })
    }

    pub fn descr(&self) -> &'static str {
        match self {
            ItemKind::ExternCrate(..) => "extern crate",
            ItemKind::Use(..) => "`use` import",
            ItemKind::Static(..) => "static item",
            ItemKind::Const(..) => "constant item",
            ItemKind::Fn { .. } => "function",
            ItemKind::Macro(..) => "macro",
            ItemKind::Mod(..) => "module",
            ItemKind::ForeignMod { .. } => "extern block",
            ItemKind::GlobalAsm { .. } => "global asm item",
            ItemKind::TyAlias(..) => "type alias",
            ItemKind::Enum(..) => "enum",
            ItemKind::Struct(..) => "struct",
            ItemKind::Union(..) => "union",
            ItemKind::Trait(..) => "trait",
            ItemKind::TraitAlias(..) => "trait alias",
            ItemKind::Impl(..) => "implementation",
        }
    }
}

/// A reference from an trait to one of its associated items. This
/// contains the item's id, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct TraitItemRef {
    pub id: TraitItemId,
    pub ident: Ident,
    pub kind: AssocItemKind,
    pub span: Span,
}

/// A reference from an impl to one of its associated items. This
/// contains the item's ID, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct ImplItemRef {
    pub id: ImplItemId,
    pub ident: Ident,
    pub kind: AssocItemKind,
    pub span: Span,
    /// When we are in a trait impl, link to the trait-item's id.
    pub trait_item_def_id: Option<DefId>,
}

#[derive(Copy, Clone, PartialEq, Debug, HashStable_Generic)]
pub enum AssocItemKind {
    Const,
    Fn { has_self: bool },
    Type,
}

// The bodies for items are stored "out of line", in a separate
// hashmap in the `Crate`. Here we just record the hir-id of the item
// so it can fetched later.
#[derive(Copy, Clone, PartialEq, Eq, Encodable, Decodable, Debug, HashStable_Generic)]
pub struct ForeignItemId {
    pub owner_id: OwnerId,
}

impl ForeignItemId {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }
}

/// A reference from a foreign block to one of its items. This
/// contains the item's ID, naturally, but also the item's name and
/// some other high-level details (like whether it is an associated
/// type or method, and whether it is public). This allows other
/// passes to find the impl they want without loading the ID (which
/// means fewer edges in the incremental compilation graph).
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct ForeignItemRef {
    pub id: ForeignItemId,
    pub ident: Ident,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub struct ForeignItem<'hir> {
    pub ident: Ident,
    pub kind: ForeignItemKind<'hir>,
    pub owner_id: OwnerId,
    pub span: Span,
    pub vis_span: Span,
}

impl ForeignItem<'_> {
    #[inline]
    pub fn hir_id(&self) -> HirId {
        // Items are always HIR owners.
        HirId::make_owner(self.owner_id.def_id)
    }

    pub fn foreign_item_id(&self) -> ForeignItemId {
        ForeignItemId { owner_id: self.owner_id }
    }
}

/// An item within an `extern` block.
#[derive(Debug, Clone, Copy, HashStable_Generic)]
pub enum ForeignItemKind<'hir> {
    /// A foreign function.
    ///
    /// All argument idents are actually always present (i.e. `Some`), but
    /// `&[Option<Ident>]` is used because of code paths shared with `TraitFn`
    /// and `BareFnTy`. The sharing is due to all of these cases not allowing
    /// arbitrary patterns for parameters.
    Fn(FnSig<'hir>, &'hir [Option<Ident>], &'hir Generics<'hir>),
    /// A foreign static item (`static ext: u8`).
    Static(&'hir Ty<'hir>, Mutability, Safety),
    /// A foreign type.
    Type,
}

/// A variable captured by a closure.
#[derive(Debug, Copy, Clone, HashStable_Generic)]
pub struct Upvar {
    /// First span where it is accessed (there can be multiple).
    pub span: Span,
}

// The TraitCandidate's import_ids is empty if the trait is defined in the same module, and
// has length > 0 if the trait is found through an chain of imports, starting with the
// import/use statement in the scope where the trait is used.
#[derive(Debug, Clone, HashStable_Generic)]
pub struct TraitCandidate {
    pub def_id: DefId,
    pub import_ids: SmallVec<[LocalDefId; 1]>,
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum OwnerNode<'hir> {
    Item(&'hir Item<'hir>),
    ForeignItem(&'hir ForeignItem<'hir>),
    TraitItem(&'hir TraitItem<'hir>),
    ImplItem(&'hir ImplItem<'hir>),
    Crate(&'hir Mod<'hir>),
    Synthetic,
}

impl<'hir> OwnerNode<'hir> {
    pub fn span(&self) -> Span {
        match self {
            OwnerNode::Item(Item { span, .. })
            | OwnerNode::ForeignItem(ForeignItem { span, .. })
            | OwnerNode::ImplItem(ImplItem { span, .. })
            | OwnerNode::TraitItem(TraitItem { span, .. }) => *span,
            OwnerNode::Crate(Mod { spans: ModSpans { inner_span, .. }, .. }) => *inner_span,
            OwnerNode::Synthetic => unreachable!(),
        }
    }

    pub fn fn_sig(self) -> Option<&'hir FnSig<'hir>> {
        match self {
            OwnerNode::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::Item(Item { kind: ItemKind::Fn { sig: fn_sig, .. }, .. })
            | OwnerNode::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Fn(fn_sig, _, _), ..
            }) => Some(fn_sig),
            _ => None,
        }
    }

    pub fn fn_decl(self) -> Option<&'hir FnDecl<'hir>> {
        match self {
            OwnerNode::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | OwnerNode::Item(Item { kind: ItemKind::Fn { sig: fn_sig, .. }, .. })
            | OwnerNode::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Fn(fn_sig, _, _), ..
            }) => Some(fn_sig.decl),
            _ => None,
        }
    }

    pub fn body_id(&self) -> Option<BodyId> {
        match self {
            OwnerNode::Item(Item {
                kind:
                    ItemKind::Static(_, _, _, body)
                    | ItemKind::Const(_, _, _, body)
                    | ItemKind::Fn { body, .. },
                ..
            })
            | OwnerNode::TraitItem(TraitItem {
                kind:
                    TraitItemKind::Fn(_, TraitFn::Provided(body)) | TraitItemKind::Const(_, Some(body)),
                ..
            })
            | OwnerNode::ImplItem(ImplItem {
                kind: ImplItemKind::Fn(_, body) | ImplItemKind::Const(_, body),
                ..
            }) => Some(*body),
            _ => None,
        }
    }

    pub fn generics(self) -> Option<&'hir Generics<'hir>> {
        Node::generics(self.into())
    }

    pub fn def_id(self) -> OwnerId {
        match self {
            OwnerNode::Item(Item { owner_id, .. })
            | OwnerNode::TraitItem(TraitItem { owner_id, .. })
            | OwnerNode::ImplItem(ImplItem { owner_id, .. })
            | OwnerNode::ForeignItem(ForeignItem { owner_id, .. }) => *owner_id,
            OwnerNode::Crate(..) => crate::CRATE_HIR_ID.owner,
            OwnerNode::Synthetic => unreachable!(),
        }
    }

    /// Check if node is an impl block.
    pub fn is_impl_block(&self) -> bool {
        matches!(self, OwnerNode::Item(Item { kind: ItemKind::Impl(_), .. }))
    }

    expect_methods_self! {
        expect_item,         &'hir Item<'hir>,        OwnerNode::Item(n),        n;
        expect_foreign_item, &'hir ForeignItem<'hir>, OwnerNode::ForeignItem(n), n;
        expect_impl_item,    &'hir ImplItem<'hir>,    OwnerNode::ImplItem(n),    n;
        expect_trait_item,   &'hir TraitItem<'hir>,   OwnerNode::TraitItem(n),   n;
    }
}

impl<'hir> From<&'hir Item<'hir>> for OwnerNode<'hir> {
    fn from(val: &'hir Item<'hir>) -> Self {
        OwnerNode::Item(val)
    }
}

impl<'hir> From<&'hir ForeignItem<'hir>> for OwnerNode<'hir> {
    fn from(val: &'hir ForeignItem<'hir>) -> Self {
        OwnerNode::ForeignItem(val)
    }
}

impl<'hir> From<&'hir ImplItem<'hir>> for OwnerNode<'hir> {
    fn from(val: &'hir ImplItem<'hir>) -> Self {
        OwnerNode::ImplItem(val)
    }
}

impl<'hir> From<&'hir TraitItem<'hir>> for OwnerNode<'hir> {
    fn from(val: &'hir TraitItem<'hir>) -> Self {
        OwnerNode::TraitItem(val)
    }
}

impl<'hir> From<OwnerNode<'hir>> for Node<'hir> {
    fn from(val: OwnerNode<'hir>) -> Self {
        match val {
            OwnerNode::Item(n) => Node::Item(n),
            OwnerNode::ForeignItem(n) => Node::ForeignItem(n),
            OwnerNode::ImplItem(n) => Node::ImplItem(n),
            OwnerNode::TraitItem(n) => Node::TraitItem(n),
            OwnerNode::Crate(n) => Node::Crate(n),
            OwnerNode::Synthetic => Node::Synthetic,
        }
    }
}

#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum Node<'hir> {
    Param(&'hir Param<'hir>),
    Item(&'hir Item<'hir>),
    ForeignItem(&'hir ForeignItem<'hir>),
    TraitItem(&'hir TraitItem<'hir>),
    ImplItem(&'hir ImplItem<'hir>),
    Variant(&'hir Variant<'hir>),
    Field(&'hir FieldDef<'hir>),
    AnonConst(&'hir AnonConst),
    ConstBlock(&'hir ConstBlock),
    ConstArg(&'hir ConstArg<'hir>),
    Expr(&'hir Expr<'hir>),
    ExprField(&'hir ExprField<'hir>),
    Stmt(&'hir Stmt<'hir>),
    PathSegment(&'hir PathSegment<'hir>),
    Ty(&'hir Ty<'hir>),
    AssocItemConstraint(&'hir AssocItemConstraint<'hir>),
    TraitRef(&'hir TraitRef<'hir>),
    OpaqueTy(&'hir OpaqueTy<'hir>),
    TyPat(&'hir TyPat<'hir>),
    Pat(&'hir Pat<'hir>),
    PatField(&'hir PatField<'hir>),
    /// Needed as its own node with its own HirId for tracking
    /// the unadjusted type of literals within patterns
    /// (e.g. byte str literals not being of slice type).
    PatExpr(&'hir PatExpr<'hir>),
    Arm(&'hir Arm<'hir>),
    Block(&'hir Block<'hir>),
    LetStmt(&'hir LetStmt<'hir>),
    /// `Ctor` refers to the constructor of an enum variant or struct. Only tuple or unit variants
    /// with synthesized constructors.
    Ctor(&'hir VariantData<'hir>),
    Lifetime(&'hir Lifetime),
    GenericParam(&'hir GenericParam<'hir>),
    Crate(&'hir Mod<'hir>),
    Infer(&'hir InferArg),
    WherePredicate(&'hir WherePredicate<'hir>),
    PreciseCapturingNonLifetimeArg(&'hir PreciseCapturingNonLifetimeArg),
    // Created by query feeding
    Synthetic,
    Err(Span),
}

impl<'hir> Node<'hir> {
    /// Get the identifier of this `Node`, if applicable.
    ///
    /// # Edge cases
    ///
    /// Calling `.ident()` on a [`Node::Ctor`] will return `None`
    /// because `Ctor`s do not have identifiers themselves.
    /// Instead, call `.ident()` on the parent struct/variant, like so:
    ///
    /// ```ignore (illustrative)
    /// ctor
    ///     .ctor_hir_id()
    ///     .map(|ctor_id| tcx.parent_hir_node(ctor_id))
    ///     .and_then(|parent| parent.ident())
    /// ```
    pub fn ident(&self) -> Option<Ident> {
        match self {
            Node::Item(item) => item.kind.ident(),
            Node::TraitItem(TraitItem { ident, .. })
            | Node::ImplItem(ImplItem { ident, .. })
            | Node::ForeignItem(ForeignItem { ident, .. })
            | Node::Field(FieldDef { ident, .. })
            | Node::Variant(Variant { ident, .. })
            | Node::PathSegment(PathSegment { ident, .. }) => Some(*ident),
            Node::Lifetime(lt) => Some(lt.ident),
            Node::GenericParam(p) => Some(p.name.ident()),
            Node::AssocItemConstraint(c) => Some(c.ident),
            Node::PatField(f) => Some(f.ident),
            Node::ExprField(f) => Some(f.ident),
            Node::PreciseCapturingNonLifetimeArg(a) => Some(a.ident),
            Node::Param(..)
            | Node::AnonConst(..)
            | Node::ConstBlock(..)
            | Node::ConstArg(..)
            | Node::Expr(..)
            | Node::Stmt(..)
            | Node::Block(..)
            | Node::Ctor(..)
            | Node::Pat(..)
            | Node::TyPat(..)
            | Node::PatExpr(..)
            | Node::Arm(..)
            | Node::LetStmt(..)
            | Node::Crate(..)
            | Node::Ty(..)
            | Node::TraitRef(..)
            | Node::OpaqueTy(..)
            | Node::Infer(..)
            | Node::WherePredicate(..)
            | Node::Synthetic
            | Node::Err(..) => None,
        }
    }

    pub fn fn_decl(self) -> Option<&'hir FnDecl<'hir>> {
        match self {
            Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | Node::Item(Item { kind: ItemKind::Fn { sig: fn_sig, .. }, .. })
            | Node::ForeignItem(ForeignItem { kind: ForeignItemKind::Fn(fn_sig, _, _), .. }) => {
                Some(fn_sig.decl)
            }
            Node::Expr(Expr { kind: ExprKind::Closure(Closure { fn_decl, .. }), .. }) => {
                Some(fn_decl)
            }
            _ => None,
        }
    }

    /// Get a `hir::Impl` if the node is an impl block for the given `trait_def_id`.
    pub fn impl_block_of_trait(self, trait_def_id: DefId) -> Option<&'hir Impl<'hir>> {
        if let Node::Item(Item { kind: ItemKind::Impl(impl_block), .. }) = self
            && let Some(trait_ref) = impl_block.of_trait
            && let Some(trait_id) = trait_ref.trait_def_id()
            && trait_id == trait_def_id
        {
            Some(impl_block)
        } else {
            None
        }
    }

    pub fn fn_sig(self) -> Option<&'hir FnSig<'hir>> {
        match self {
            Node::TraitItem(TraitItem { kind: TraitItemKind::Fn(fn_sig, _), .. })
            | Node::ImplItem(ImplItem { kind: ImplItemKind::Fn(fn_sig, _), .. })
            | Node::Item(Item { kind: ItemKind::Fn { sig: fn_sig, .. }, .. })
            | Node::ForeignItem(ForeignItem { kind: ForeignItemKind::Fn(fn_sig, _, _), .. }) => {
                Some(fn_sig)
            }
            _ => None,
        }
    }

    /// Get the type for constants, assoc types, type aliases and statics.
    pub fn ty(self) -> Option<&'hir Ty<'hir>> {
        match self {
            Node::Item(it) => match it.kind {
                ItemKind::TyAlias(_, ty, _)
                | ItemKind::Static(_, ty, _, _)
                | ItemKind::Const(_, ty, _, _) => Some(ty),
                ItemKind::Impl(impl_item) => Some(&impl_item.self_ty),
                _ => None,
            },
            Node::TraitItem(it) => match it.kind {
                TraitItemKind::Const(ty, _) => Some(ty),
                TraitItemKind::Type(_, ty) => ty,
                _ => None,
            },
            Node::ImplItem(it) => match it.kind {
                ImplItemKind::Const(ty, _) => Some(ty),
                ImplItemKind::Type(ty) => Some(ty),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn alias_ty(self) -> Option<&'hir Ty<'hir>> {
        match self {
            Node::Item(Item { kind: ItemKind::TyAlias(_, ty, _), .. }) => Some(ty),
            _ => None,
        }
    }

    #[inline]
    pub fn associated_body(&self) -> Option<(LocalDefId, BodyId)> {
        match self {
            Node::Item(Item {
                owner_id,
                kind:
                    ItemKind::Const(_, _, _, body)
                    | ItemKind::Static(.., body)
                    | ItemKind::Fn { body, .. },
                ..
            })
            | Node::TraitItem(TraitItem {
                owner_id,
                kind:
                    TraitItemKind::Const(_, Some(body)) | TraitItemKind::Fn(_, TraitFn::Provided(body)),
                ..
            })
            | Node::ImplItem(ImplItem {
                owner_id,
                kind: ImplItemKind::Const(_, body) | ImplItemKind::Fn(_, body),
                ..
            }) => Some((owner_id.def_id, *body)),

            Node::Item(Item {
                owner_id, kind: ItemKind::GlobalAsm { asm: _, fake_body }, ..
            }) => Some((owner_id.def_id, *fake_body)),

            Node::Expr(Expr { kind: ExprKind::Closure(Closure { def_id, body, .. }), .. }) => {
                Some((*def_id, *body))
            }

            Node::AnonConst(constant) => Some((constant.def_id, constant.body)),
            Node::ConstBlock(constant) => Some((constant.def_id, constant.body)),

            _ => None,
        }
    }

    pub fn body_id(&self) -> Option<BodyId> {
        Some(self.associated_body()?.1)
    }

    pub fn generics(self) -> Option<&'hir Generics<'hir>> {
        match self {
            Node::ForeignItem(ForeignItem {
                kind: ForeignItemKind::Fn(_, _, generics), ..
            })
            | Node::TraitItem(TraitItem { generics, .. })
            | Node::ImplItem(ImplItem { generics, .. }) => Some(generics),
            Node::Item(item) => item.kind.generics(),
            _ => None,
        }
    }

    pub fn as_owner(self) -> Option<OwnerNode<'hir>> {
        match self {
            Node::Item(i) => Some(OwnerNode::Item(i)),
            Node::ForeignItem(i) => Some(OwnerNode::ForeignItem(i)),
            Node::TraitItem(i) => Some(OwnerNode::TraitItem(i)),
            Node::ImplItem(i) => Some(OwnerNode::ImplItem(i)),
            Node::Crate(i) => Some(OwnerNode::Crate(i)),
            Node::Synthetic => Some(OwnerNode::Synthetic),
            _ => None,
        }
    }

    pub fn fn_kind(self) -> Option<FnKind<'hir>> {
        match self {
            Node::Item(i) => match i.kind {
                ItemKind::Fn { ident, sig, generics, .. } => {
                    Some(FnKind::ItemFn(ident, generics, sig.header))
                }
                _ => None,
            },
            Node::TraitItem(ti) => match ti.kind {
                TraitItemKind::Fn(ref sig, _) => Some(FnKind::Method(ti.ident, sig)),
                _ => None,
            },
            Node::ImplItem(ii) => match ii.kind {
                ImplItemKind::Fn(ref sig, _) => Some(FnKind::Method(ii.ident, sig)),
                _ => None,
            },
            Node::Expr(e) => match e.kind {
                ExprKind::Closure { .. } => Some(FnKind::Closure),
                _ => None,
            },
            _ => None,
        }
    }

    expect_methods_self! {
        expect_param,         &'hir Param<'hir>,        Node::Param(n),        n;
        expect_item,          &'hir Item<'hir>,         Node::Item(n),         n;
        expect_foreign_item,  &'hir ForeignItem<'hir>,  Node::ForeignItem(n),  n;
        expect_trait_item,    &'hir TraitItem<'hir>,    Node::TraitItem(n),    n;
        expect_impl_item,     &'hir ImplItem<'hir>,     Node::ImplItem(n),     n;
        expect_variant,       &'hir Variant<'hir>,      Node::Variant(n),      n;
        expect_field,         &'hir FieldDef<'hir>,     Node::Field(n),        n;
        expect_anon_const,    &'hir AnonConst,          Node::AnonConst(n),    n;
        expect_inline_const,  &'hir ConstBlock,         Node::ConstBlock(n),   n;
        expect_expr,          &'hir Expr<'hir>,         Node::Expr(n),         n;
        expect_expr_field,    &'hir ExprField<'hir>,    Node::ExprField(n),    n;
        expect_stmt,          &'hir Stmt<'hir>,         Node::Stmt(n),         n;
        expect_path_segment,  &'hir PathSegment<'hir>,  Node::PathSegment(n),  n;
        expect_ty,            &'hir Ty<'hir>,           Node::Ty(n),           n;
        expect_assoc_item_constraint,  &'hir AssocItemConstraint<'hir>,  Node::AssocItemConstraint(n),  n;
        expect_trait_ref,     &'hir TraitRef<'hir>,     Node::TraitRef(n),     n;
        expect_opaque_ty,     &'hir OpaqueTy<'hir>,     Node::OpaqueTy(n),     n;
        expect_pat,           &'hir Pat<'hir>,          Node::Pat(n),          n;
        expect_pat_field,     &'hir PatField<'hir>,     Node::PatField(n),     n;
        expect_arm,           &'hir Arm<'hir>,          Node::Arm(n),          n;
        expect_block,         &'hir Block<'hir>,        Node::Block(n),        n;
        expect_let_stmt,      &'hir LetStmt<'hir>,      Node::LetStmt(n),      n;
        expect_ctor,          &'hir VariantData<'hir>,  Node::Ctor(n),         n;
        expect_lifetime,      &'hir Lifetime,           Node::Lifetime(n),     n;
        expect_generic_param, &'hir GenericParam<'hir>, Node::GenericParam(n), n;
        expect_crate,         &'hir Mod<'hir>,          Node::Crate(n),        n;
        expect_infer,         &'hir InferArg,           Node::Infer(n),        n;
        expect_closure,       &'hir Closure<'hir>, Node::Expr(Expr { kind: ExprKind::Closure(n), .. }), n;
    }
}

// Some nodes are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(Block<'_>, 48);
    static_assert_size!(Body<'_>, 24);
    static_assert_size!(Expr<'_>, 64);
    static_assert_size!(ExprKind<'_>, 48);
    static_assert_size!(FnDecl<'_>, 40);
    static_assert_size!(ForeignItem<'_>, 88);
    static_assert_size!(ForeignItemKind<'_>, 56);
    static_assert_size!(GenericArg<'_>, 16);
    static_assert_size!(GenericBound<'_>, 64);
    static_assert_size!(Generics<'_>, 56);
    static_assert_size!(Impl<'_>, 80);
    static_assert_size!(ImplItem<'_>, 88);
    static_assert_size!(ImplItemKind<'_>, 40);
    static_assert_size!(Item<'_>, 88);
    static_assert_size!(ItemKind<'_>, 64);
    static_assert_size!(LetStmt<'_>, 72);
    static_assert_size!(Param<'_>, 32);
    static_assert_size!(Pat<'_>, 72);
    static_assert_size!(Path<'_>, 40);
    static_assert_size!(PathSegment<'_>, 48);
    static_assert_size!(PatKind<'_>, 48);
    static_assert_size!(QPath<'_>, 24);
    static_assert_size!(Res, 12);
    static_assert_size!(Stmt<'_>, 32);
    static_assert_size!(StmtKind<'_>, 16);
    static_assert_size!(TraitItem<'_>, 88);
    static_assert_size!(TraitItemKind<'_>, 48);
    static_assert_size!(Ty<'_>, 48);
    static_assert_size!(TyKind<'_>, 32);
    // tidy-alphabetical-end
}

#[cfg(test)]
mod tests;
