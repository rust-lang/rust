use reexport::*;
use rustc::hir;
use rustc::hir::*;
use rustc::hir::intravisit::{FnKind, Visitor, walk_ty, NestedVisitorMap};
use rustc::lint::*;
use rustc::ty::{self, Ty};
use rustc::ty::subst::Substs;
use std::cmp::Ordering;
use syntax::ast::{IntTy, UintTy, FloatTy};
use syntax::attr::IntType;
use syntax::codemap::Span;
use utils::{comparisons, higher, in_external_macro, in_macro, match_def_path, snippet, span_help_and_lint, span_lint,
            span_lint_and_sugg, opt_def_id, last_path_segment, type_size};
use utils::paths;

/// Handles all the linting of funky types
#[allow(missing_copy_implementations)]
pub struct TypePass;

/// **What it does:** Checks for use of `Box<Vec<_>>` anywhere in the code.
///
/// **Why is this bad?** `Vec` already keeps its contents in a separate area on
/// the heap. So if you `Box` it, you just add another level of indirection
/// without any benefit whatsoever.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// struct X {
///     values: Box<Vec<Foo>>,
/// }
/// ```
declare_lint! {
    pub BOX_VEC,
    Warn,
    "usage of `Box<Vec<T>>`, vector elements are already on the heap"
}

/// **What it does:** Checks for usage of any `LinkedList`, suggesting to use a
/// `Vec` or a `VecDeque` (formerly called `RingBuf`).
///
/// **Why is this bad?** Gankro says:
///
/// > The TL;DR of `LinkedList` is that it's built on a massive amount of pointers and indirection.
/// > It wastes memory, it has terrible cache locality, and is all-around slow. `RingBuf`, while
/// > "only" amortized for push/pop, should be faster in the general case for almost every possible
/// > workload, and isn't even amortized at all if you can predict the capacity you need.
/// >
/// > `LinkedList`s are only really good if you're doing a lot of merging or splitting of lists.
/// > This is because they can just mangle some pointers instead of actually copying the data. Even
/// > if you're doing a lot of insertion in the middle of the list, `RingBuf` can still be better
/// > because of how expensive it is to seek to the middle of a `LinkedList`.
///
/// **Known problems:** False positives – the instances where using a
/// `LinkedList` makes sense are few and far between, but they can still happen.
///
/// **Example:**
/// ```rust
/// let x = LinkedList::new();
/// ```
declare_lint! {
    pub LINKEDLIST,
    Warn,
    "usage of LinkedList, usually a vector is faster, or a more specialized data \
     structure like a VecDeque"
}

/// **What it does:** Checks for use of `&Box<T>` anywhere in the code.
///
/// **Why is this bad?** Any `&Box<T>` can also be a `&T`, which is more general.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn foo(bar: &Box<T>) { ... }
/// ```
declare_lint! {
    pub BORROWED_BOX,
    Warn,
    "a borrow of a boxed type"
}

impl LintPass for TypePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(BOX_VEC, LINKEDLIST, BORROWED_BOX)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypePass {
    fn check_fn(&mut self, cx: &LateContext, _: FnKind, decl: &FnDecl, _: &Body, _: Span, id: NodeId) {
        // skip trait implementations, see #605
        if let Some(map::NodeItem(item)) = cx.tcx.hir.find(cx.tcx.hir.get_parent(id)) {
            if let ItemImpl(_, _, _, _, Some(..), _, _) = item.node {
                return;
            }
        }

        check_fn_decl(cx, decl);
    }

    fn check_struct_field(&mut self, cx: &LateContext, field: &StructField) {
        check_ty(cx, &field.ty, false);
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &TraitItem) {
        match item.node {
            TraitItemKind::Const(ref ty, _) |
            TraitItemKind::Type(_, Some(ref ty)) => check_ty(cx, ty, false),
            TraitItemKind::Method(ref sig, _) => check_fn_decl(cx, &sig.decl),
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext, local: &Local) {
        if let Some(ref ty) = local.ty {
            check_ty(cx, ty, true);
        }
    }
}

fn check_fn_decl(cx: &LateContext, decl: &FnDecl) {
    for input in &decl.inputs {
        check_ty(cx, input, false);
    }

    if let FunctionRetTy::Return(ref ty) = decl.output {
        check_ty(cx, ty, false);
    }
}

/// Recursively check for `TypePass` lints in the given type. Stop at the first
/// lint found.
///
/// The parameter `is_local` distinguishes the context of the type; types from
/// local bindings should only be checked for the `BORROWED_BOX` lint.
fn check_ty(cx: &LateContext, ast_ty: &hir::Ty, is_local: bool) {
    if in_macro(ast_ty.span) {
        return;
    }
    match ast_ty.node {
        TyPath(ref qpath) if !is_local => {
            let def = cx.tables.qpath_def(qpath, ast_ty.id);
            if let Some(def_id) = opt_def_id(def) {
                if Some(def_id) == cx.tcx.lang_items.owned_box() {
                    let last = last_path_segment(qpath);
                    if_let_chain! {[
                        let PathParameters::AngleBracketedParameters(ref ag) = last.parameters,
                        let Some(vec) = ag.types.get(0),
                        let TyPath(ref qpath) = vec.node,
                        let def::Def::Struct(..) = cx.tables.qpath_def(qpath, vec.id),
                        let Some(did) = opt_def_id(cx.tables.qpath_def(qpath, vec.id)),
                        match_def_path(cx.tcx, did, &paths::VEC),
                    ], {
                        span_help_and_lint(cx,
                                           BOX_VEC,
                                           ast_ty.span,
                                           "you seem to be trying to use `Box<Vec<T>>`. Consider using just `Vec<T>`",
                                           "`Vec<T>` is already on the heap, `Box<Vec<T>>` makes an extra allocation.");
                        return; // don't recurse into the type
                    }}
                } else if match_def_path(cx.tcx, def_id, &paths::LINKED_LIST) {
                    span_help_and_lint(cx,
                                       LINKEDLIST,
                                       ast_ty.span,
                                       "I see you're using a LinkedList! Perhaps you meant some other data structure?",
                                       "a VecDeque might work");
                    return; // don't recurse into the type
                }
            }
            match *qpath {
                QPath::Resolved(Some(ref ty), ref p) => {
                    check_ty(cx, ty, is_local);
                    for ty in p.segments.iter().flat_map(|seg| seg.parameters.types()) {
                        check_ty(cx, ty, is_local);
                    }
                },
                QPath::Resolved(None, ref p) => {
                    for ty in p.segments.iter().flat_map(|seg| seg.parameters.types()) {
                        check_ty(cx, ty, is_local);
                    }
                },
                QPath::TypeRelative(ref ty, ref seg) => {
                    check_ty(cx, ty, is_local);
                    for ty in seg.parameters.types() {
                        check_ty(cx, ty, is_local);
                    }
                },
            }
        },
        TyRptr(ref lt, MutTy { ref ty, ref mutbl }) => {
            match ty.node {
                TyPath(ref qpath) => {
                    let def = cx.tables.qpath_def(qpath, ast_ty.id);
                    if_let_chain! {[
                        let Some(def_id) = opt_def_id(def),
                        Some(def_id) == cx.tcx.lang_items.owned_box(),
                        let QPath::Resolved(None, ref path) = *qpath,
                        let [ref bx] = *path.segments,
                        let PathParameters::AngleBracketedParameters(ref ab_data) = bx.parameters,
                        let [ref inner] = *ab_data.types
                    ], {
                        let ltopt = if lt.is_elided() {
                            "".to_owned()
                        } else {
                            format!("{} ", lt.name.as_str())
                        };
                        let mutopt = if *mutbl == Mutability::MutMutable {
                            "mut "
                        } else {
                            ""
                        };
                        span_lint_and_sugg(cx,
                            BORROWED_BOX,
                            ast_ty.span,
                            "you seem to be trying to use `&Box<T>`. Consider using just `&T`",
                            "try",
                            format!("&{}{}{}", ltopt, mutopt, &snippet(cx, inner.span, ".."))
                        );
                        return; // don't recurse into the type
                    }};
                    check_ty(cx, ty, is_local);
                },
                _ => check_ty(cx, ty, is_local),
            }
        },
        // recurse
        TySlice(ref ty) |
        TyArray(ref ty, _) |
        TyPtr(MutTy { ref ty, .. }) => check_ty(cx, ty, is_local),
        TyTup(ref tys) => {
            for ty in tys {
                check_ty(cx, ty, is_local);
            }
        },
        _ => {},
    }
}

#[allow(missing_copy_implementations)]
pub struct LetPass;

/// **What it does:** Checks for binding a unit value.
///
/// **Why is this bad?** A unit value cannot usefully be used anywhere. So
/// binding one is kind of pointless.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let x = { 1; };
/// ```
declare_lint! {
    pub LET_UNIT_VALUE,
    Warn,
    "creating a let binding to a value of unit type, which usually can't be used afterwards"
}

fn check_let_unit(cx: &LateContext, decl: &Decl) {
    if let DeclLocal(ref local) = decl.node {
        match cx.tables.pat_ty(&local.pat).sty {
            ty::TyTuple(slice, _) if slice.is_empty() => {
                if in_external_macro(cx, decl.span) || in_macro(local.pat.span) {
                    return;
                }
                if higher::is_from_for_desugar(decl) {
                    return;
                }
                span_lint(cx,
                          LET_UNIT_VALUE,
                          decl.span,
                          &format!("this let-binding has unit value. Consider omitting `let {} =`",
                                   snippet(cx, local.pat.span, "..")));
            },
            _ => (),
        }
    }
}

impl LintPass for LetPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(LET_UNIT_VALUE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for LetPass {
    fn check_decl(&mut self, cx: &LateContext<'a, 'tcx>, decl: &'tcx Decl) {
        check_let_unit(cx, decl)
    }
}

/// **What it does:** Checks for comparisons to unit.
///
/// **Why is this bad?** Unit is always equal to itself, and thus is just a
/// clumsily written constant. Mostly this happens when someone accidentally
/// adds semicolons at the end of the operands.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// if { foo(); } == { bar(); } { baz(); }
/// ```
/// is equal to
/// ```rust
/// { foo(); bar(); baz(); }
/// ```
declare_lint! {
    pub UNIT_CMP,
    Warn,
    "comparing unit values"
}

#[allow(missing_copy_implementations)]
pub struct UnitCmp;

impl LintPass for UnitCmp {
    fn get_lints(&self) -> LintArray {
        lint_array!(UNIT_CMP)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnitCmp {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if in_macro(expr.span) {
            return;
        }
        if let ExprBinary(ref cmp, ref left, _) = expr.node {
            let op = cmp.node;
            if op.is_comparison() {
                match cx.tables.expr_ty(left).sty {
                    ty::TyTuple(slice, _) if slice.is_empty() => {
                        let result = match op {
                            BiEq | BiLe | BiGe => "true",
                            _ => "false",
                        };
                        span_lint(cx,
                                  UNIT_CMP,
                                  expr.span,
                                  &format!("{}-comparison of unit values detected. This will always be {}",
                                           op.as_str(),
                                           result));
                    },
                    _ => (),
                }
            }
        }
    }
}

pub struct CastPass;

/// **What it does:** Checks for casts from any numerical to a float type where
/// the receiving type cannot store all values from the original type without
/// rounding errors. This possible rounding is to be expected, so this lint is
/// `Allow` by default.
///
/// Basically, this warns on casting any integer with 32 or more bits to `f32`
/// or any 64-bit integer to `f64`.
///
/// **Why is this bad?** It's not bad at all. But in some applications it can be
/// helpful to know where precision loss can take place. This lint can help find
/// those places in the code.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let x = u64::MAX; x as f64
/// ```
declare_lint! {
    pub CAST_PRECISION_LOSS,
    Allow,
    "casts that cause loss of precision, e.g. `x as f32` where `x: u64`"
}

/// **What it does:** Checks for casts from a signed to an unsigned numerical
/// type. In this case, negative values wrap around to large positive values,
/// which can be quite surprising in practice. However, as the cast works as
/// defined, this lint is `Allow` by default.
///
/// **Why is this bad?** Possibly surprising results. You can activate this lint
/// as a one-time check to see where numerical wrapping can arise.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let y: i8 = -1;
/// y as u128  // will return 18446744073709551615
/// ```
declare_lint! {
    pub CAST_SIGN_LOSS,
    Allow,
    "casts from signed types to unsigned types, e.g. `x as u32` where `x: i32`"
}

/// **What it does:** Checks for on casts between numerical types that may
/// truncate large values. This is expected behavior, so the cast is `Allow` by
/// default.
///
/// **Why is this bad?** In some problem domains, it is good practice to avoid
/// truncation. This lint can be activated to help assess where additional
/// checks could be beneficial.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// fn as_u8(x: u64) -> u8 { x as u8 }
/// ```
declare_lint! {
    pub CAST_POSSIBLE_TRUNCATION,
    Allow,
    "casts that may cause truncation of the value, e.g. `x as u8` where `x: u32`, \
     or `x as i32` where `x: f32`"
}

/// **What it does:** Checks for casts from an unsigned type to a signed type of
/// the same size. Performing such a cast is a 'no-op' for the compiler,
/// i.e. nothing is changed at the bit level, and the binary representation of
/// the value is reinterpreted. This can cause wrapping if the value is too big
/// for the target signed type. However, the cast works as defined, so this lint
/// is `Allow` by default.
///
/// **Why is this bad?** While such a cast is not bad in itself, the results can
/// be surprising when this is not the intended behavior, as demonstrated by the
/// example below.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// u32::MAX as i32  // will yield a value of `-1`
/// ```
declare_lint! {
    pub CAST_POSSIBLE_WRAP,
    Allow,
    "casts that may cause wrapping around the value, e.g. `x as i32` where `x: u32` \
     and `x > i32::MAX`"
}

/// **What it does:** Checks for casts to the same type.
///
/// **Why is this bad?** It's just unnecessary.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// let _ = 2i32 as i32
/// ```
declare_lint! {
    pub UNNECESSARY_CAST,
    Warn,
    "cast to the same type, e.g. `x as i32` where `x: i32`"
}

/// Returns the size in bits of an integral type.
/// Will return 0 if the type is not an int or uint variant
fn int_ty_to_nbits(typ: Ty) -> usize {
    let n = match typ.sty {
        ty::TyInt(i) => 4 << (i as usize),
        ty::TyUint(u) => 4 << (u as usize),
        _ => 0,
    };
    // n == 4 is the usize/isize case
    if n == 4 {
        ::std::mem::size_of::<usize>() * 8
    } else {
        n
    }
}

fn is_isize_or_usize(typ: Ty) -> bool {
    match typ.sty {
        ty::TyInt(IntTy::Is) |
        ty::TyUint(UintTy::Us) => true,
        _ => false,
    }
}

fn span_precision_loss_lint(cx: &LateContext, expr: &Expr, cast_from: Ty, cast_to_f64: bool) {
    let mantissa_nbits = if cast_to_f64 { 52 } else { 23 };
    let arch_dependent = is_isize_or_usize(cast_from) && cast_to_f64;
    let arch_dependent_str = "on targets with 64-bit wide pointers ";
    let from_nbits_str = if arch_dependent {
        "64".to_owned()
    } else if is_isize_or_usize(cast_from) {
        "32 or 64".to_owned()
    } else {
        int_ty_to_nbits(cast_from).to_string()
    };
    span_lint(cx,
              CAST_PRECISION_LOSS,
              expr.span,
              &format!("casting {0} to {1} causes a loss of precision {2}({0} is {3} bits wide, but {1}'s mantissa \
                        is only {4} bits wide)",
                       cast_from,
                       if cast_to_f64 { "f64" } else { "f32" },
                       if arch_dependent {
                           arch_dependent_str
                       } else {
                           ""
                       },
                       from_nbits_str,
                       mantissa_nbits));
}

enum ArchSuffix {
    _32,
    _64,
    None,
}

fn check_truncation_and_wrapping(cx: &LateContext, expr: &Expr, cast_from: Ty, cast_to: Ty) {
    let arch_64_suffix = " on targets with 64-bit wide pointers";
    let arch_32_suffix = " on targets with 32-bit wide pointers";
    let cast_unsigned_to_signed = !cast_from.is_signed() && cast_to.is_signed();
    let (from_nbits, to_nbits) = (int_ty_to_nbits(cast_from), int_ty_to_nbits(cast_to));
    let (span_truncation, suffix_truncation, span_wrap, suffix_wrap) = match (is_isize_or_usize(cast_from),
                                                                              is_isize_or_usize(cast_to)) {
        (true, true) | (false, false) => {
            (to_nbits < from_nbits,
             ArchSuffix::None,
             to_nbits == from_nbits && cast_unsigned_to_signed,
             ArchSuffix::None)
        },
        (true, false) => {
            (to_nbits <= 32,
             if to_nbits == 32 {
                 ArchSuffix::_64
             } else {
                 ArchSuffix::None
             },
             to_nbits <= 32 && cast_unsigned_to_signed,
             ArchSuffix::_32)
        },
        (false, true) => {
            (from_nbits == 64,
             ArchSuffix::_32,
             cast_unsigned_to_signed,
             if from_nbits == 64 {
                 ArchSuffix::_64
             } else {
                 ArchSuffix::_32
             })
        },
    };
    if span_truncation {
        span_lint(cx,
                  CAST_POSSIBLE_TRUNCATION,
                  expr.span,
                  &format!("casting {} to {} may truncate the value{}",
                           cast_from,
                           cast_to,
                           match suffix_truncation {
                               ArchSuffix::_32 => arch_32_suffix,
                               ArchSuffix::_64 => arch_64_suffix,
                               ArchSuffix::None => "",
                           }));
    }
    if span_wrap {
        span_lint(cx,
                  CAST_POSSIBLE_WRAP,
                  expr.span,
                  &format!("casting {} to {} may wrap around the value{}",
                           cast_from,
                           cast_to,
                           match suffix_wrap {
                               ArchSuffix::_32 => arch_32_suffix,
                               ArchSuffix::_64 => arch_64_suffix,
                               ArchSuffix::None => "",
                           }));
    }
}

impl LintPass for CastPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CAST_PRECISION_LOSS,
                    CAST_SIGN_LOSS,
                    CAST_POSSIBLE_TRUNCATION,
                    CAST_POSSIBLE_WRAP,
                    UNNECESSARY_CAST)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CastPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprCast(ref ex, _) = expr.node {
            let (cast_from, cast_to) = (cx.tables.expr_ty(ex), cx.tables.expr_ty(expr));
            if let ExprLit(ref lit) = ex.node {
                use syntax::ast::{LitKind, LitIntType};
                match lit.node {
                    LitKind::Int(_, LitIntType::Unsuffixed) |
                    LitKind::FloatUnsuffixed(_) => {},
                    _ => {
                        if cast_from.sty == cast_to.sty && !in_external_macro(cx, expr.span) {
                            span_lint(cx,
                                      UNNECESSARY_CAST,
                                      expr.span,
                                      &format!("casting to the same type is unnecessary (`{}` -> `{}`)",
                                               cast_from,
                                               cast_to));
                        }
                    },
                }
            }
            if cast_from.is_numeric() && cast_to.is_numeric() && !in_external_macro(cx, expr.span) {
                match (cast_from.is_integral(), cast_to.is_integral()) {
                    (true, false) => {
                        let from_nbits = int_ty_to_nbits(cast_from);
                        let to_nbits = if let ty::TyFloat(FloatTy::F32) = cast_to.sty {
                            32
                        } else {
                            64
                        };
                        if is_isize_or_usize(cast_from) || from_nbits >= to_nbits {
                            span_precision_loss_lint(cx, expr, cast_from, to_nbits == 64);
                        }
                    },
                    (false, true) => {
                        span_lint(cx,
                                  CAST_POSSIBLE_TRUNCATION,
                                  expr.span,
                                  &format!("casting {} to {} may truncate the value", cast_from, cast_to));
                        if !cast_to.is_signed() {
                            span_lint(cx,
                                      CAST_SIGN_LOSS,
                                      expr.span,
                                      &format!("casting {} to {} may lose the sign of the value", cast_from, cast_to));
                        }
                    },
                    (true, true) => {
                        if cast_from.is_signed() && !cast_to.is_signed() {
                            span_lint(cx,
                                      CAST_SIGN_LOSS,
                                      expr.span,
                                      &format!("casting {} to {} may lose the sign of the value", cast_from, cast_to));
                        }
                        check_truncation_and_wrapping(cx, expr, cast_from, cast_to);
                    },
                    (false, false) => {
                        if let (&ty::TyFloat(FloatTy::F64), &ty::TyFloat(FloatTy::F32)) =
                            (&cast_from.sty, &cast_to.sty) {
                            span_lint(cx,
                                      CAST_POSSIBLE_TRUNCATION,
                                      expr.span,
                                      "casting f64 to f32 may truncate the value");
                        }
                    },
                }
            }
        }
    }
}

/// **What it does:** Checks for types used in structs, parameters and `let`
/// declarations above a certain complexity threshold.
///
/// **Why is this bad?** Too complex types make the code less readable. Consider
/// using a `type` definition to simplify them.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// struct Foo { inner: Rc<Vec<Vec<Box<(u32, u32, u32, u32)>>>> }
/// ```
declare_lint! {
    pub TYPE_COMPLEXITY,
    Warn,
    "usage of very complex types that might be better factored into `type` definitions"
}

#[allow(missing_copy_implementations)]
pub struct TypeComplexityPass {
    threshold: u64,
}

impl TypeComplexityPass {
    pub fn new(threshold: u64) -> Self {
        TypeComplexityPass { threshold: threshold }
    }
}

impl LintPass for TypeComplexityPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TYPE_COMPLEXITY)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for TypeComplexityPass {
    fn check_fn(
        &mut self,
        cx: &LateContext<'a, 'tcx>,
        _: FnKind<'tcx>,
        decl: &'tcx FnDecl,
        _: &'tcx Body,
        _: Span,
        _: NodeId
    ) {
        self.check_fndecl(cx, decl);
    }

    fn check_struct_field(&mut self, cx: &LateContext<'a, 'tcx>, field: &'tcx StructField) {
        // enum variants are also struct fields now
        self.check_type(cx, &field.ty);
    }

    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        match item.node {
            ItemStatic(ref ty, _, _) |
            ItemConst(ref ty, _) => self.check_type(cx, ty),
            // functions, enums, structs, impls and traits are covered
            _ => (),
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx TraitItem) {
        match item.node {
            TraitItemKind::Const(ref ty, _) |
            TraitItemKind::Type(_, Some(ref ty)) => self.check_type(cx, ty),
            TraitItemKind::Method(MethodSig { ref decl, .. }, TraitMethod::Required(_)) => self.check_fndecl(cx, decl),
            // methods with default impl are covered by check_fn
            _ => (),
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx ImplItem) {
        match item.node {
            ImplItemKind::Const(ref ty, _) |
            ImplItemKind::Type(ref ty) => self.check_type(cx, ty),
            // methods are covered by check_fn
            _ => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'a, 'tcx>, local: &'tcx Local) {
        if let Some(ref ty) = local.ty {
            self.check_type(cx, ty);
        }
    }
}

impl<'a, 'tcx> TypeComplexityPass {
    fn check_fndecl(&self, cx: &LateContext<'a, 'tcx>, decl: &'tcx FnDecl) {
        for arg in &decl.inputs {
            self.check_type(cx, arg);
        }
        if let Return(ref ty) = decl.output {
            self.check_type(cx, ty);
        }
    }

    fn check_type(&self, cx: &LateContext, ty: &hir::Ty) {
        if in_macro(ty.span) {
            return;
        }
        let score = {
            let mut visitor = TypeComplexityVisitor {
                score: 0,
                nest: 1,
            };
            visitor.visit_ty(ty);
            visitor.score
        };

        if score > self.threshold {
            span_lint(cx,
                      TYPE_COMPLEXITY,
                      ty.span,
                      "very complex type used. Consider factoring parts into `type` definitions");
        }
    }
}

/// Walks a type and assigns a complexity score to it.
struct TypeComplexityVisitor {
    /// total complexity score of the type
    score: u64,
    /// current nesting level
    nest: u64,
}

impl<'tcx> Visitor<'tcx> for TypeComplexityVisitor {
    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        let (add_score, sub_nest) = match ty.node {
            // _, &x and *x have only small overhead; don't mess with nesting level
            TyInfer | TyPtr(..) | TyRptr(..) => (1, 0),

            // the "normal" components of a type: named types, arrays/tuples
            TyPath(..) | TySlice(..) | TyTup(..) | TyArray(..) => (10 * self.nest, 1),

            // function types bring a lot of overhead
            TyBareFn(..) => (50 * self.nest, 1),

            TyTraitObject(ref param_bounds, _) => {
                let has_lifetime_parameters = param_bounds.iter()
                    .any(|bound| !bound.bound_lifetimes.is_empty());
                if has_lifetime_parameters {
                    // complex trait bounds like A<'a, 'b>
                    (50 * self.nest, 1)
                } else {
                    // simple trait bounds like A + B
                    (20 * self.nest, 0)
                }
            },

            _ => (0, 0),
        };
        self.score += add_score;
        self.nest += sub_nest;
        walk_ty(self, ty);
        self.nest -= sub_nest;
    }
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

/// **What it does:** Checks for expressions where a character literal is cast
/// to `u8` and suggests using a byte literal instead.
///
/// **Why is this bad?** In general, casting values to smaller types is
/// error-prone and should be avoided where possible. In the particular case of
/// converting a character literal to u8, it is easy to avoid by just using a
/// byte literal instead. As an added bonus, `b'a'` is even slightly shorter
/// than `'a' as u8`.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// 'x' as u8
/// ```
declare_lint! {
    pub CHAR_LIT_AS_U8,
    Warn,
    "casting a character literal to u8"
}

pub struct CharLitAsU8;

impl LintPass for CharLitAsU8 {
    fn get_lints(&self) -> LintArray {
        lint_array!(CHAR_LIT_AS_U8)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for CharLitAsU8 {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        use syntax::ast::{LitKind, UintTy};

        if let ExprCast(ref e, _) = expr.node {
            if let ExprLit(ref l) = e.node {
                if let LitKind::Char(_) = l.node {
                    if ty::TyUint(UintTy::U8) == cx.tables.expr_ty(expr).sty && !in_macro(expr.span) {
                        let msg = "casting character literal to u8. `char`s \
                                   are 4 bytes wide in rust, so casting to u8 \
                                   truncates them";
                        let help = format!("Consider using a byte literal instead:\nb{}", snippet(cx, e.span, "'x'"));
                        span_help_and_lint(cx, CHAR_LIT_AS_U8, expr.span, msg, &help);
                    }
                }
            }
        }
    }
}

/// **What it does:** Checks for comparisons where one side of the relation is
/// either the minimum or maximum value for its type and warns if it involves a
/// case that is always true or always false. Only integer and boolean types are
/// checked.
///
/// **Why is this bad?** An expression like `min <= x` may misleadingly imply
/// that is is possible for `x` to be less than the minimum. Expressions like
/// `max < x` are probably mistakes.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// vec.len() <= 0
/// 100 > std::i32::MAX
/// ```
declare_lint! {
    pub ABSURD_EXTREME_COMPARISONS,
    Warn,
    "a comparison with a maximum or minimum value that is always true or false"
}

pub struct AbsurdExtremeComparisons;

impl LintPass for AbsurdExtremeComparisons {
    fn get_lints(&self) -> LintArray {
        lint_array!(ABSURD_EXTREME_COMPARISONS)
    }
}

enum ExtremeType {
    Minimum,
    Maximum,
}

struct ExtremeExpr<'a> {
    which: ExtremeType,
    expr: &'a Expr,
}

enum AbsurdComparisonResult {
    AlwaysFalse,
    AlwaysTrue,
    InequalityImpossible,
}



fn detect_absurd_comparison<'a>(
    cx: &LateContext,
    op: BinOp_,
    lhs: &'a Expr,
    rhs: &'a Expr
) -> Option<(ExtremeExpr<'a>, AbsurdComparisonResult)> {
    use types::ExtremeType::*;
    use types::AbsurdComparisonResult::*;
    use utils::comparisons::*;

    // absurd comparison only makes sense on primitive types
    // primitive types don't implement comparison operators with each other
    if cx.tables.expr_ty(lhs) != cx.tables.expr_ty(rhs) {
        return None;
    }

    let normalized = normalize_comparison(op, lhs, rhs);
    let (rel, normalized_lhs, normalized_rhs) = if let Some(val) = normalized {
        val
    } else {
        return None;
    };

    let lx = detect_extreme_expr(cx, normalized_lhs);
    let rx = detect_extreme_expr(cx, normalized_rhs);

    Some(match rel {
        Rel::Lt => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, AlwaysFalse), // max < x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, AlwaysFalse), // x < min
                _ => return None,
            }
        },
        Rel::Le => {
            match (lx, rx) {
                (Some(l @ ExtremeExpr { which: Minimum, .. }), _) => (l, AlwaysTrue), // min <= x
                (Some(l @ ExtremeExpr { which: Maximum, .. }), _) => (l, InequalityImpossible), //max <= x
                (_, Some(r @ ExtremeExpr { which: Minimum, .. })) => (r, InequalityImpossible), // x <= min
                (_, Some(r @ ExtremeExpr { which: Maximum, .. })) => (r, AlwaysTrue), // x <= max
                _ => return None,
            }
        },
        Rel::Ne | Rel::Eq => return None,
    })
}

fn detect_extreme_expr<'a>(cx: &LateContext, expr: &'a Expr) -> Option<ExtremeExpr<'a>> {
    use rustc::middle::const_val::ConstVal::*;
    use rustc_const_math::*;
    use rustc_const_eval::*;
    use types::ExtremeType::*;

    let ty = cx.tables.expr_ty(expr);

    match ty.sty {
        ty::TyBool | ty::TyInt(_) | ty::TyUint(_) => (),
        _ => return None,
    };

    let parent_item = cx.tcx.hir.get_parent(expr.id);
    let parent_def_id = cx.tcx.hir.local_def_id(parent_item);
    let substs = Substs::identity_for_item(cx.tcx, parent_def_id);
    let cv = match ConstContext::new(cx.tcx, cx.param_env.and(substs), cx.tables).eval(expr) {
        Ok(val) => val,
        Err(_) => return None,
    };

    let which = match (&ty.sty, cv) {
        (&ty::TyBool, Bool(false)) |
        (&ty::TyInt(IntTy::Is), Integral(Isize(Is32(::std::i32::MIN)))) |
        (&ty::TyInt(IntTy::Is), Integral(Isize(Is64(::std::i64::MIN)))) |
        (&ty::TyInt(IntTy::I8), Integral(I8(::std::i8::MIN))) |
        (&ty::TyInt(IntTy::I16), Integral(I16(::std::i16::MIN))) |
        (&ty::TyInt(IntTy::I32), Integral(I32(::std::i32::MIN))) |
        (&ty::TyInt(IntTy::I64), Integral(I64(::std::i64::MIN))) |
        (&ty::TyInt(IntTy::I128), Integral(I128(::std::i128::MIN))) |
        (&ty::TyUint(UintTy::Us), Integral(Usize(Us32(::std::u32::MIN)))) |
        (&ty::TyUint(UintTy::Us), Integral(Usize(Us64(::std::u64::MIN)))) |
        (&ty::TyUint(UintTy::U8), Integral(U8(::std::u8::MIN))) |
        (&ty::TyUint(UintTy::U16), Integral(U16(::std::u16::MIN))) |
        (&ty::TyUint(UintTy::U32), Integral(U32(::std::u32::MIN))) |
        (&ty::TyUint(UintTy::U64), Integral(U64(::std::u64::MIN))) |
        (&ty::TyUint(UintTy::U128), Integral(U128(::std::u128::MIN))) => Minimum,

        (&ty::TyBool, Bool(true)) |
        (&ty::TyInt(IntTy::Is), Integral(Isize(Is32(::std::i32::MAX)))) |
        (&ty::TyInt(IntTy::Is), Integral(Isize(Is64(::std::i64::MAX)))) |
        (&ty::TyInt(IntTy::I8), Integral(I8(::std::i8::MAX))) |
        (&ty::TyInt(IntTy::I16), Integral(I16(::std::i16::MAX))) |
        (&ty::TyInt(IntTy::I32), Integral(I32(::std::i32::MAX))) |
        (&ty::TyInt(IntTy::I64), Integral(I64(::std::i64::MAX))) |
        (&ty::TyInt(IntTy::I128), Integral(I128(::std::i128::MAX))) |
        (&ty::TyUint(UintTy::Us), Integral(Usize(Us32(::std::u32::MAX)))) |
        (&ty::TyUint(UintTy::Us), Integral(Usize(Us64(::std::u64::MAX)))) |
        (&ty::TyUint(UintTy::U8), Integral(U8(::std::u8::MAX))) |
        (&ty::TyUint(UintTy::U16), Integral(U16(::std::u16::MAX))) |
        (&ty::TyUint(UintTy::U32), Integral(U32(::std::u32::MAX))) |
        (&ty::TyUint(UintTy::U64), Integral(U64(::std::u64::MAX))) |
        (&ty::TyUint(UintTy::U128), Integral(U128(::std::u128::MAX))) => Maximum,

        _ => return None,
    };
    Some(ExtremeExpr {
        which: which,
        expr: expr,
    })
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for AbsurdExtremeComparisons {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        use types::ExtremeType::*;
        use types::AbsurdComparisonResult::*;

        if let ExprBinary(ref cmp, ref lhs, ref rhs) = expr.node {
            if let Some((culprit, result)) = detect_absurd_comparison(cx, cmp.node, lhs, rhs) {
                if !in_macro(expr.span) {
                    let msg = "this comparison involving the minimum or maximum element for this \
                               type contains a case that is always true or always false";

                    let conclusion = match result {
                        AlwaysFalse => "this comparison is always false".to_owned(),
                        AlwaysTrue => "this comparison is always true".to_owned(),
                        InequalityImpossible => {
                            format!("the case where the two sides are not equal never occurs, consider using {} == {} \
                                     instead",
                                    snippet(cx, lhs.span, "lhs"),
                                    snippet(cx, rhs.span, "rhs"))
                        },
                    };

                    let help = format!("because {} is the {} value for this type, {}",
                                       snippet(cx, culprit.expr.span, "x"),
                                       match culprit.which {
                                           Minimum => "minimum",
                                           Maximum => "maximum",
                                       },
                                       conclusion);

                    span_help_and_lint(cx, ABSURD_EXTREME_COMPARISONS, expr.span, msg, &help);
                }
            }
        }
    }
}

/// **What it does:** Checks for comparisons where the relation is always either
/// true or false, but where one side has been upcast so that the comparison is
/// necessary. Only integer types are checked.
///
/// **Why is this bad?** An expression like `let x : u8 = ...; (x as u32) > 300`
/// will mistakenly imply that it is possible for `x` to be outside the range of
/// `u8`.
///
/// **Known problems:** https://github.com/rust-lang-nursery/rust-clippy/issues/886
///
/// **Example:**
/// ```rust
/// let x : u8 = ...; (x as u32) > 300
/// ```
declare_lint! {
    pub INVALID_UPCAST_COMPARISONS,
    Allow,
    "a comparison involving an upcast which is always true or false"
}

pub struct InvalidUpcastComparisons;

impl LintPass for InvalidUpcastComparisons {
    fn get_lints(&self) -> LintArray {
        lint_array!(INVALID_UPCAST_COMPARISONS)
    }
}

#[derive(Copy, Clone, Debug, Eq)]
enum FullInt {
    S(i128),
    U(u128),
}

impl FullInt {
    #[allow(cast_sign_loss)]
    fn cmp_s_u(s: i128, u: u128) -> Ordering {
        if s < 0 {
            Ordering::Less
        } else if u > (i128::max_value() as u128) {
            Ordering::Greater
        } else {
            (s as u128).cmp(&u)
        }
    }
}

impl PartialEq for FullInt {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other).expect("partial_cmp only returns Some(_)") == Ordering::Equal
    }
}

impl PartialOrd for FullInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(match (self, other) {
            (&FullInt::S(s), &FullInt::S(o)) => s.cmp(&o),
            (&FullInt::U(s), &FullInt::U(o)) => s.cmp(&o),
            (&FullInt::S(s), &FullInt::U(o)) => Self::cmp_s_u(s, o),
            (&FullInt::U(s), &FullInt::S(o)) => Self::cmp_s_u(o, s).reverse(),
        })
    }
}
impl Ord for FullInt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).expect("partial_cmp for FullInt can never return None")
    }
}


fn numeric_cast_precast_bounds<'a>(cx: &LateContext, expr: &'a Expr) -> Option<(FullInt, FullInt)> {
    use syntax::ast::{IntTy, UintTy};
    use std::*;

    if let ExprCast(ref cast_exp, _) = expr.node {
        let pre_cast_ty = cx.tables.expr_ty(cast_exp);
        let cast_ty = cx.tables.expr_ty(expr);
        // if it's a cast from i32 to u32 wrapping will invalidate all these checks
        if type_size(cx, pre_cast_ty) == type_size(cx, cast_ty) {
            return None;
        }
        match pre_cast_ty.sty {
            ty::TyInt(int_ty) => {
                Some(match int_ty {
                    IntTy::I8 => (FullInt::S(i8::min_value() as i128), FullInt::S(i8::max_value() as i128)),
                    IntTy::I16 => (FullInt::S(i16::min_value() as i128), FullInt::S(i16::max_value() as i128)),
                    IntTy::I32 => (FullInt::S(i32::min_value() as i128), FullInt::S(i32::max_value() as i128)),
                    IntTy::I64 => (FullInt::S(i64::min_value() as i128), FullInt::S(i64::max_value() as i128)),
                    IntTy::I128 => (FullInt::S(i128::min_value() as i128), FullInt::S(i128::max_value() as i128)),
                    IntTy::Is => (FullInt::S(isize::min_value() as i128), FullInt::S(isize::max_value() as i128)),
                })
            },
            ty::TyUint(uint_ty) => {
                Some(match uint_ty {
                    UintTy::U8 => (FullInt::U(u8::min_value() as u128), FullInt::U(u8::max_value() as u128)),
                    UintTy::U16 => (FullInt::U(u16::min_value() as u128), FullInt::U(u16::max_value() as u128)),
                    UintTy::U32 => (FullInt::U(u32::min_value() as u128), FullInt::U(u32::max_value() as u128)),
                    UintTy::U64 => (FullInt::U(u64::min_value() as u128), FullInt::U(u64::max_value() as u128)),
                    UintTy::U128 => (FullInt::U(u128::min_value() as u128), FullInt::U(u128::max_value() as u128)),
                    UintTy::Us => (FullInt::U(usize::min_value() as u128), FullInt::U(usize::max_value() as u128)),
                })
            },
            _ => None,
        }
    } else {
        None
    }
}

#[allow(cast_possible_wrap)]
fn node_as_const_fullint(cx: &LateContext, expr: &Expr) -> Option<FullInt> {
    use rustc::middle::const_val::ConstVal::*;
    use rustc_const_eval::ConstContext;

    let parent_item = cx.tcx.hir.get_parent(expr.id);
    let parent_def_id = cx.tcx.hir.local_def_id(parent_item);
    let substs = Substs::identity_for_item(cx.tcx, parent_def_id);
    match ConstContext::new(cx.tcx, cx.param_env.and(substs), cx.tables).eval(expr) {
        Ok(val) => {
            if let Integral(const_int) = val {
                match const_int.int_type() {
                    IntType::SignedInt(_) => Some(FullInt::S(const_int.to_u128_unchecked() as i128)),
                    IntType::UnsignedInt(_) => Some(FullInt::U(const_int.to_u128_unchecked())),
                }
            } else {
                None
            }
        },
        Err(_) => None,
    }
}

fn err_upcast_comparison(cx: &LateContext, span: &Span, expr: &Expr, always: bool) {
    if let ExprCast(ref cast_val, _) = expr.node {
        span_lint(cx,
                  INVALID_UPCAST_COMPARISONS,
                  *span,
                  &format!(
                "because of the numeric bounds on `{}` prior to casting, this expression is always {}",
                snippet(cx, cast_val.span, "the expression"),
                if always { "true" } else { "false" },
            ));
    }
}

fn upcast_comparison_bounds_err(
    cx: &LateContext,
    span: &Span,
    rel: comparisons::Rel,
    lhs_bounds: Option<(FullInt, FullInt)>,
    lhs: &Expr,
    rhs: &Expr,
    invert: bool
) {
    use utils::comparisons::*;

    if let Some((lb, ub)) = lhs_bounds {
        if let Some(norm_rhs_val) = node_as_const_fullint(cx, rhs) {
            if rel == Rel::Eq || rel == Rel::Ne {
                if norm_rhs_val < lb || norm_rhs_val > ub {
                    err_upcast_comparison(cx, span, lhs, rel == Rel::Ne);
                }
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val < lb
                    } else {
                        ub < norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val <= lb
                    } else {
                        ub <= norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, true)
            } else if match rel {
                Rel::Lt => {
                    if invert {
                        norm_rhs_val >= ub
                    } else {
                        lb >= norm_rhs_val
                    }
                },
                Rel::Le => {
                    if invert {
                        norm_rhs_val > ub
                    } else {
                        lb > norm_rhs_val
                    }
                },
                Rel::Eq | Rel::Ne => unreachable!(),
            } {
                err_upcast_comparison(cx, span, lhs, false)
            }
        }
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for InvalidUpcastComparisons {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprBinary(ref cmp, ref lhs, ref rhs) = expr.node {

            let normalized = comparisons::normalize_comparison(cmp.node, lhs, rhs);
            let (rel, normalized_lhs, normalized_rhs) = if let Some(val) = normalized {
                val
            } else {
                return;
            };

            let lhs_bounds = numeric_cast_precast_bounds(cx, normalized_lhs);
            let rhs_bounds = numeric_cast_precast_bounds(cx, normalized_rhs);

            upcast_comparison_bounds_err(cx, &expr.span, rel, lhs_bounds, normalized_lhs, normalized_rhs, false);
            upcast_comparison_bounds_err(cx, &expr.span, rel, rhs_bounds, normalized_rhs, normalized_lhs, true);
        }
    }
}
