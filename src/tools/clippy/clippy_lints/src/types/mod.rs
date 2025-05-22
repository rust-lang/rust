mod borrowed_box;
mod box_collection;
mod linked_list;
mod option_option;
mod owned_cow;
mod rc_buffer;
mod rc_mutex;
mod redundant_allocation;
mod type_complexity;
mod utils;
mod vec_box;

use clippy_config::Conf;
use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Body, FnDecl, FnRetTy, GenericArg, ImplItem, ImplItemKind, Item, ItemKind, LetStmt, MutTy, QPath, TraitFn,
    TraitItem, TraitItemKind, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::LocalDefId;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Box<T>` where T is a collection such as Vec anywhere in the code.
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
    ///
    /// ### Why is this bad?
    /// Collections already keeps their contents in a separate area on
    /// the heap. So if you `Box` them, you just add another level of indirection
    /// without any benefit whatsoever.
    ///
    /// ### Example
    /// ```rust,ignore
    /// struct X {
    ///     values: Box<Vec<Foo>>,
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// struct X {
    ///     values: Vec<Foo>,
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub BOX_COLLECTION,
    perf,
    "usage of `Box<Vec<T>>`, vector elements are already on the heap"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Vec<Box<T>>` where T: Sized anywhere in the code.
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
    ///
    /// ### Why is this bad?
    /// `Vec` already keeps its contents in a separate area on
    /// the heap. So if you `Box` its contents, you just add another level of indirection.
    ///
    /// ### Example
    /// ```no_run
    /// struct X {
    ///     values: Vec<Box<i32>>,
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```no_run
    /// struct X {
    ///     values: Vec<i32>,
    /// }
    /// ```
    #[clippy::version = "1.33.0"]
    pub VEC_BOX,
    complexity,
    "usage of `Vec<Box<T>>` where T: Sized, vector elements are already on the heap"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `Option<Option<_>>` in function signatures and type
    /// definitions
    ///
    /// ### Why is this bad?
    /// `Option<_>` represents an optional value. `Option<Option<_>>`
    /// represents an optional value which itself wraps an optional. This is logically the
    /// same thing as an optional value but has an unneeded extra level of wrapping.
    ///
    /// If you have a case where `Some(Some(_))`, `Some(None)` and `None` are distinct cases,
    /// consider a custom `enum` instead, with clear names for each case.
    ///
    /// ### Example
    /// ```no_run
    /// fn get_data() -> Option<Option<u32>> {
    ///     None
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```no_run
    /// pub enum Contents {
    ///     Data(Vec<u8>), // Was Some(Some(Vec<u8>))
    ///     NotYetFetched, // Was Some(None)
    ///     None,          // Was None
    /// }
    ///
    /// fn get_data() -> Contents {
    ///     Contents::None
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub OPTION_OPTION,
    pedantic,
    "usage of `Option<Option<T>>`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of any `LinkedList`, suggesting to use a
    /// `Vec` or a `VecDeque` (formerly called `RingBuf`).
    ///
    /// ### Why is this bad?
    /// Gankra says:
    ///
    /// > The TL;DR of `LinkedList` is that it's built on a massive amount of
    /// pointers and indirection.
    /// > It wastes memory, it has terrible cache locality, and is all-around slow.
    /// `RingBuf`, while
    /// > "only" amortized for push/pop, should be faster in the general case for
    /// almost every possible
    /// > workload, and isn't even amortized at all if you can predict the capacity
    /// you need.
    /// >
    /// > `LinkedList`s are only really good if you're doing a lot of merging or
    /// splitting of lists.
    /// > This is because they can just mangle some pointers instead of actually
    /// copying the data. Even
    /// > if you're doing a lot of insertion in the middle of the list, `RingBuf`
    /// can still be better
    /// > because of how expensive it is to seek to the middle of a `LinkedList`.
    ///
    /// ### Known problems
    /// False positives – the instances where using a
    /// `LinkedList` makes sense are few and far between, but they can still happen.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::collections::LinkedList;
    /// let x: LinkedList<usize> = LinkedList::new();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LINKEDLIST,
    pedantic,
    "usage of LinkedList, usually a vector is faster, or a more specialized data structure like a `VecDeque`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `&Box<T>` anywhere in the code.
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
    ///
    /// ### Why is this bad?
    /// A `&Box<T>` parameter requires the function caller to box `T` first before passing it to a function.
    /// Using `&T` defines a concrete type for the parameter and generalizes the function, this would also
    /// auto-deref to `&T` at the function call site if passed a `&Box<T>`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// fn foo(bar: &Box<T>) { ... }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// fn foo(bar: &T) { ... }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub BORROWED_BOX,
    complexity,
    "a borrow of a boxed type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of redundant allocations anywhere in the code.
    ///
    /// ### Why is this bad?
    /// Expressions such as `Rc<&T>`, `Rc<Rc<T>>`, `Rc<Arc<T>>`, `Rc<Box<T>>`, `Arc<&T>`, `Arc<Rc<T>>`,
    /// `Arc<Arc<T>>`, `Arc<Box<T>>`, `Box<&T>`, `Box<Rc<T>>`, `Box<Arc<T>>`, `Box<Box<T>>`, add an unnecessary level of indirection.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::rc::Rc;
    /// fn foo(bar: Rc<&usize>) {}
    /// ```
    ///
    /// Better:
    ///
    /// ```no_run
    /// fn foo(bar: &usize) {}
    /// ```
    #[clippy::version = "1.44.0"]
    pub REDUNDANT_ALLOCATION,
    perf,
    "redundant allocation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Rc<T>` and `Arc<T>` when `T` is a mutable buffer type such as `String` or `Vec`.
    ///
    /// ### Why restrict this?
    /// Expressions such as `Rc<String>` usually have no advantage over `Rc<str>`, since
    /// it is larger and involves an extra level of indirection, and doesn't implement `Borrow<str>`.
    ///
    /// While mutating a buffer type would still be possible with `Rc::get_mut()`, it only
    /// works if there are no additional references yet, which usually defeats the purpose of
    /// enclosing it in a shared ownership type. Instead, additionally wrapping the inner
    /// type with an interior mutable container (such as `RefCell` or `Mutex`) would normally
    /// be used.
    ///
    /// ### Known problems
    /// This pattern can be desirable to avoid the overhead of a `RefCell` or `Mutex` for
    /// cases where mutation only happens before there are any additional references.
    ///
    /// ### Example
    /// ```rust,ignore
    /// # use std::rc::Rc;
    /// fn foo(interned: Rc<String>) { ... }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// fn foo(interned: Rc<str>) { ... }
    /// ```
    #[clippy::version = "1.48.0"]
    pub RC_BUFFER,
    restriction,
    "shared ownership of a buffer type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for types used in structs, parameters and `let`
    /// declarations above a certain complexity threshold.
    ///
    /// ### Why is this bad?
    /// Too complex types make the code less readable. Consider
    /// using a `type` definition to simplify them.
    ///
    /// ### Example
    /// ```no_run
    /// # use std::rc::Rc;
    /// struct PointMatrixContainer {
    ///     matrix: Rc<Vec<Vec<Box<(u32, u32, u32, u32)>>>>,
    /// }
    ///
    /// fn main() {
    ///     let point_matrix: Vec<Vec<Box<(u32, u32, u32, u32)>>> = vec![
    ///         vec![
    ///             Box::new((1, 2, 3, 4)),
    ///             Box::new((5, 6, 7, 8)),
    ///         ],
    ///         vec![
    ///             Box::new((9, 10, 11, 12)),
    ///         ],
    ///     ];
    ///
    ///     let shared_point_matrix: Rc<Vec<Vec<Box<(u32, u32, u32, u32)>>>> = Rc::new(point_matrix);
    ///
    ///     let container = PointMatrixContainer {
    ///         matrix: shared_point_matrix,
    ///     };
    ///
    ///     // ...
    /// }
    /// ```
    /// Use instead:
    /// ### Example
    /// ```no_run
    /// # use std::rc::Rc;
    /// type PointMatrix = Vec<Vec<Box<(u32, u32, u32, u32)>>>;
    /// type SharedPointMatrix = Rc<PointMatrix>;
    ///
    /// struct PointMatrixContainer {
    ///     matrix: SharedPointMatrix,
    /// }
    ///
    /// fn main() {
    ///     let point_matrix: PointMatrix = vec![
    ///         vec![
    ///             Box::new((1, 2, 3, 4)),
    ///             Box::new((5, 6, 7, 8)),
    ///         ],
    ///         vec![
    ///             Box::new((9, 10, 11, 12)),
    ///         ],
    ///     ];
    ///
    ///     let shared_point_matrix: SharedPointMatrix = Rc::new(point_matrix);
    ///
    ///     let container = PointMatrixContainer {
    ///         matrix: shared_point_matrix,
    ///     };
    ///
    ///     // ...
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TYPE_COMPLEXITY,
    complexity,
    "usage of very complex types that might be better factored into `type` definitions"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `Rc<Mutex<T>>`.
    ///
    /// ### Why restrict this?
    /// `Rc` is used in single thread and `Mutex` is used in multi thread.
    /// Consider using `Rc<RefCell<T>>` in single thread or `Arc<Mutex<T>>` in multi thread.
    ///
    /// ### Known problems
    /// Sometimes combining generic types can lead to the requirement that a
    /// type use Rc in conjunction with Mutex. We must consider those cases false positives, but
    /// alas they are quite hard to rule out. Luckily they are also rare.
    ///
    /// ### Example
    /// ```rust,ignore
    /// use std::rc::Rc;
    /// use std::sync::Mutex;
    /// fn foo(interned: Rc<Mutex<i32>>) { ... }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust,ignore
    /// use std::rc::Rc;
    /// use std::cell::RefCell
    /// fn foo(interned: Rc<RefCell<i32>>) { ... }
    /// ```
    #[clippy::version = "1.55.0"]
    pub RC_MUTEX,
    restriction,
    "usage of `Rc<Mutex<T>>`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Detects needlessly owned `Cow` types.
    ///
    /// ### Why is this bad?
    /// The borrowed types are usually more flexible, in that e.g. a
    /// `Cow<'_, str>` can accept both `&str` and `String` while
    /// `Cow<'_, String>` can only accept `&String` and `String`. In
    /// particular, `&str` is more general, because it allows for string
    /// literals while `&String` can only be borrowed from a heap-owned
    /// `String`).
    ///
    /// ### Known Problems
    /// The lint does not check for usage of the type. There may be external
    /// interfaces that require the use of an owned type.
    ///
    /// At least the `CString` type also has a different API than `CStr`: The
    /// former has an `as_bytes` method which the latter calls `to_bytes`.
    /// There is no guarantee that other types won't gain additional methods
    /// leading to a similar mismatch.
    ///
    /// In addition, the lint only checks for the known problematic types
    /// `String`, `Vec<_>`, `CString`, `OsString` and `PathBuf`. Custom types
    /// that implement `ToOwned` will not be detected.
    ///
    /// ### Example
    /// ```no_run
    /// let wrogn: std::borrow::Cow<'_, Vec<u8>>;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let right: std::borrow::Cow<'_, [u8]>;
    /// ```
    #[clippy::version = "1.87.0"]
    pub OWNED_COW,
    style,
    "needlessly owned Cow type"
}

pub struct Types {
    vec_box_size_threshold: u64,
    type_complexity_threshold: u64,
    avoid_breaking_exported_api: bool,
}

impl_lint_pass!(Types => [
    BOX_COLLECTION,
    VEC_BOX,
    OPTION_OPTION,
    LINKEDLIST,
    BORROWED_BOX,
    REDUNDANT_ALLOCATION,
    RC_BUFFER,
    RC_MUTEX,
    TYPE_COMPLEXITY,
    OWNED_COW
]);

impl<'tcx> LateLintPass<'tcx> for Types {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'_>,
        decl: &FnDecl<'tcx>,
        _: &Body<'_>,
        _: Span,
        def_id: LocalDefId,
    ) {
        let is_in_trait_impl = if let hir::Node::Item(item) = cx
            .tcx
            .hir_node_by_def_id(cx.tcx.hir_get_parent_item(cx.tcx.local_def_id_to_hir_id(def_id)).def_id)
        {
            matches!(item.kind, ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }))
        } else {
            false
        };

        let is_exported = cx.effective_visibilities.is_exported(def_id);

        self.check_fn_decl(
            cx,
            decl,
            CheckTyContext {
                is_in_trait_impl,
                in_body: matches!(fn_kind, FnKind::Closure),
                is_exported,
                ..CheckTyContext::default()
            },
        );
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        let is_exported = cx.effective_visibilities.is_exported(item.owner_id.def_id);

        match item.kind {
            ItemKind::Static(_, ty, _, _) | ItemKind::Const(_, ty, _, _) => self.check_ty(
                cx,
                ty,
                CheckTyContext {
                    is_exported,
                    ..CheckTyContext::default()
                },
            ),
            // functions, enums, structs, impls and traits are covered
            _ => (),
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        match item.kind {
            ImplItemKind::Const(ty, _) => {
                let is_in_trait_impl = if let hir::Node::Item(item) = cx
                    .tcx
                    .hir_node_by_def_id(cx.tcx.hir_get_parent_item(item.hir_id()).def_id)
                {
                    matches!(item.kind, ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }))
                } else {
                    false
                };

                self.check_ty(
                    cx,
                    ty,
                    CheckTyContext {
                        is_in_trait_impl,
                        ..CheckTyContext::default()
                    },
                );
            },
            // Methods are covered by check_fn.
            // Type aliases are ignored because oftentimes it's impossible to
            // make type alias declaration in trait simpler, see #1013
            ImplItemKind::Fn(..) | ImplItemKind::Type(..) => (),
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'tcx>, field: &hir::FieldDef<'tcx>) {
        if field.span.from_expansion() {
            return;
        }

        let is_exported = cx.effective_visibilities.is_exported(field.def_id);

        self.check_ty(
            cx,
            field.ty,
            CheckTyContext {
                is_exported,
                ..CheckTyContext::default()
            },
        );
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &TraitItem<'tcx>) {
        let is_exported = cx.effective_visibilities.is_exported(item.owner_id.def_id);

        let context = CheckTyContext {
            is_exported,
            ..CheckTyContext::default()
        };

        match item.kind {
            TraitItemKind::Const(ty, _) | TraitItemKind::Type(_, Some(ty)) => {
                self.check_ty(cx, ty, context);
            },
            TraitItemKind::Fn(ref sig, trait_method) => {
                // Check only methods without body
                // Methods with body are covered by check_fn.
                if let TraitFn::Required(_) = trait_method {
                    self.check_fn_decl(cx, sig.decl, context);
                }
            },
            TraitItemKind::Type(..) => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'tcx>, local: &LetStmt<'tcx>) {
        if let Some(ty) = local.ty {
            self.check_ty(
                cx,
                ty,
                CheckTyContext {
                    in_body: true,
                    ..CheckTyContext::default()
                },
            );
        }
    }
}

impl Types {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            vec_box_size_threshold: conf.vec_box_size_threshold,
            type_complexity_threshold: conf.type_complexity_threshold,
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
        }
    }

    fn check_fn_decl<'tcx>(&mut self, cx: &LateContext<'tcx>, decl: &FnDecl<'tcx>, context: CheckTyContext) {
        // Ignore functions in trait implementations as they are usually forced by the trait definition.
        //
        // FIXME: ideally we would like to warn *if the complicated type can be simplified*, but it's hard
        // to check.
        if context.is_in_trait_impl {
            return;
        }

        for input in decl.inputs {
            self.check_ty(cx, input, context);
        }

        if let FnRetTy::Return(ty) = decl.output {
            self.check_ty(cx, ty, context);
        }
    }

    /// Recursively check for `TypePass` lints in the given type. Stop at the first
    /// lint found.
    ///
    /// The parameter `is_local` distinguishes the context of the type.
    fn check_ty<'tcx>(&mut self, cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'tcx>, mut context: CheckTyContext) {
        if hir_ty.span.from_expansion() {
            return;
        }

        // Skip trait implementations; see issue #605.
        if context.is_in_trait_impl {
            return;
        }

        if !context.is_nested_call && type_complexity::check(cx, hir_ty, self.type_complexity_threshold) {
            return;
        }

        match hir_ty.kind {
            TyKind::Path(ref qpath) if !context.in_body => {
                let hir_id = hir_ty.hir_id;
                let res = cx.qpath_res(qpath, hir_id);
                if let Some(def_id) = res.opt_def_id()
                    && self.is_type_change_allowed(context)
                {
                    // All lints that are being checked in this block are guarded by
                    // the `avoid_breaking_exported_api` configuration. When adding a
                    // new lint, please also add the name to the configuration documentation
                    // in `clippy_config::conf`

                    let mut triggered = false;
                    triggered |= box_collection::check(cx, hir_ty, qpath, def_id);
                    triggered |= redundant_allocation::check(cx, hir_ty, qpath, def_id);
                    triggered |= rc_buffer::check(cx, hir_ty, qpath, def_id);
                    triggered |= vec_box::check(cx, hir_ty, qpath, def_id, self.vec_box_size_threshold);
                    triggered |= option_option::check(cx, hir_ty, qpath, def_id);
                    triggered |= linked_list::check(cx, hir_ty, def_id);
                    triggered |= rc_mutex::check(cx, hir_ty, qpath, def_id);
                    triggered |= owned_cow::check(cx, qpath, def_id);

                    if triggered {
                        return;
                    }
                }
                match *qpath {
                    QPath::Resolved(Some(ty), p) => {
                        context.is_nested_call = true;
                        self.check_ty(cx, ty, context);
                        for ty in p.segments.iter().flat_map(|seg| {
                            seg.args
                                .as_ref()
                                .map_or_else(|| [].iter(), |params| params.args.iter())
                                .filter_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                })
                        }) {
                            self.check_ty(cx, ty.as_unambig_ty(), context);
                        }
                    },
                    QPath::Resolved(None, p) => {
                        context.is_nested_call = true;
                        for ty in p.segments.iter().flat_map(|seg| {
                            seg.args
                                .as_ref()
                                .map_or_else(|| [].iter(), |params| params.args.iter())
                                .filter_map(|arg| match arg {
                                    GenericArg::Type(ty) => Some(ty),
                                    _ => None,
                                })
                        }) {
                            self.check_ty(cx, ty.as_unambig_ty(), context);
                        }
                    },
                    QPath::TypeRelative(ty, seg) => {
                        context.is_nested_call = true;
                        self.check_ty(cx, ty, context);
                        if let Some(params) = seg.args {
                            for ty in params.args.iter().filter_map(|arg| match arg {
                                GenericArg::Type(ty) => Some(ty),
                                _ => None,
                            }) {
                                self.check_ty(cx, ty.as_unambig_ty(), context);
                            }
                        }
                    },
                    QPath::LangItem(..) => {},
                }
            },
            TyKind::Path(ref qpath) => {
                let res = cx.qpath_res(qpath, hir_ty.hir_id);
                if let Some(def_id) = res.opt_def_id()
                    && self.is_type_change_allowed(context)
                {
                    owned_cow::check(cx, qpath, def_id);
                }
            },
            TyKind::Ref(lt, ref mut_ty) => {
                context.is_nested_call = true;
                if !borrowed_box::check(cx, hir_ty, lt, mut_ty) {
                    self.check_ty(cx, mut_ty.ty, context);
                }
            },
            TyKind::Slice(ty) | TyKind::Array(ty, _) | TyKind::Ptr(MutTy { ty, .. }) => {
                context.is_nested_call = true;
                self.check_ty(cx, ty, context);
            },
            TyKind::Tup(tys) => {
                context.is_nested_call = true;
                for ty in tys {
                    self.check_ty(cx, ty, context);
                }
            },
            _ => {},
        }
    }

    /// This function checks if the type is allowed to change in the current context
    /// based on the `avoid_breaking_exported_api` configuration
    fn is_type_change_allowed(&self, context: CheckTyContext) -> bool {
        !(context.is_exported && self.avoid_breaking_exported_api)
    }
}

#[allow(clippy::struct_excessive_bools, clippy::struct_field_names)]
#[derive(Clone, Copy, Default)]
struct CheckTyContext {
    is_in_trait_impl: bool,
    /// `true` for types on local variables and in closure signatures.
    in_body: bool,
    /// `true` for types that are part of the public API.
    is_exported: bool,
    is_nested_call: bool,
}
