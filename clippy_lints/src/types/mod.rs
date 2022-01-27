mod borrowed_box;
mod box_collection;
mod linked_list;
mod option_option;
mod rc_buffer;
mod rc_mutex;
mod redundant_allocation;
mod type_complexity;
mod utils;
mod vec_box;

use rustc_hir as hir;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{
    Body, FnDecl, FnRetTy, GenericArg, HirId, ImplItem, ImplItemKind, Item, ItemKind, Local, MutTy, QPath, TraitItem,
    TraitItemKind, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `Box<T>` where T is a collection such as Vec anywhere in the code.
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
    /// Checks for use of `Vec<Box<T>>` where T: Sized anywhere in the code.
    /// Check the [Box documentation](https://doc.rust-lang.org/std/boxed/index.html) for more information.
    ///
    /// ### Why is this bad?
    /// `Vec` already keeps its contents in a separate area on
    /// the heap. So if you `Box` its contents, you just add another level of indirection.
    ///
    /// ### Known problems
    /// Vec<Box<T: Sized>> makes sense if T is a large type (see [#3530](https://github.com/rust-lang/rust-clippy/issues/3530),
    /// 1st comment).
    ///
    /// ### Example
    /// ```rust
    /// struct X {
    ///     values: Vec<Box<i32>>,
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
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
    /// Checks for use of `Option<Option<_>>` in function signatures and type
    /// definitions
    ///
    /// ### Why is this bad?
    /// `Option<_>` represents an optional value. `Option<Option<_>>`
    /// represents an optional optional value which is logically the same thing as an optional
    /// value but has an unneeded extra level of wrapping.
    ///
    /// If you have a case where `Some(Some(_))`, `Some(None)` and `None` are distinct cases,
    /// consider a custom `enum` instead, with clear names for each case.
    ///
    /// ### Example
    /// ```rust
    /// fn get_data() -> Option<Option<u32>> {
    ///     None
    /// }
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
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
    /// Gankro says:
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
    /// ```rust
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
    /// Checks for use of `&Box<T>` anywhere in the code.
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
    /// Checks for use of redundant allocations anywhere in the code.
    ///
    /// ### Why is this bad?
    /// Expressions such as `Rc<&T>`, `Rc<Rc<T>>`, `Rc<Arc<T>>`, `Rc<Box<T>>`, `Arc<&T>`, `Arc<Rc<T>>`,
    /// `Arc<Arc<T>>`, `Arc<Box<T>>`, `Box<&T>`, `Box<Rc<T>>`, `Box<Arc<T>>`, `Box<Box<T>>`, add an unnecessary level of indirection.
    ///
    /// ### Example
    /// ```rust
    /// # use std::rc::Rc;
    /// fn foo(bar: Rc<&usize>) {}
    /// ```
    ///
    /// Better:
    ///
    /// ```rust
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
    /// ### Why is this bad?
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
    /// ```rust
    /// # use std::rc::Rc;
    /// struct Foo {
    ///     inner: Rc<Vec<Vec<Box<(u32, u32, u32, u32)>>>>,
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
    /// ### Why is this bad?
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

pub struct Types {
    vec_box_size_threshold: u64,
    type_complexity_threshold: u64,
    avoid_breaking_exported_api: bool,
}

impl_lint_pass!(Types => [BOX_COLLECTION, VEC_BOX, OPTION_OPTION, LINKEDLIST, BORROWED_BOX, REDUNDANT_ALLOCATION, RC_BUFFER, RC_MUTEX, TYPE_COMPLEXITY]);

impl<'tcx> LateLintPass<'tcx> for Types {
    fn check_fn(&mut self, cx: &LateContext<'_>, _: FnKind<'_>, decl: &FnDecl<'_>, _: &Body<'_>, _: Span, id: HirId) {
        let is_in_trait_impl =
            if let Some(hir::Node::Item(item)) = cx.tcx.hir().find_by_def_id(cx.tcx.hir().get_parent_item(id)) {
                matches!(item.kind, ItemKind::Impl(hir::Impl { of_trait: Some(_), .. }))
            } else {
                false
            };

        let is_exported = cx.access_levels.is_exported(cx.tcx.hir().local_def_id(id));

        self.check_fn_decl(
            cx,
            decl,
            CheckTyContext {
                is_in_trait_impl,
                is_exported,
                ..CheckTyContext::default()
            },
        );
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let is_exported = cx.access_levels.is_exported(item.def_id);

        match item.kind {
            ItemKind::Static(ty, _, _) | ItemKind::Const(ty, _) => self.check_ty(
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

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        match item.kind {
            ImplItemKind::Const(ty, _) => {
                let is_in_trait_impl = if let Some(hir::Node::Item(item)) =
                    cx.tcx.hir().find_by_def_id(cx.tcx.hir().get_parent_item(item.hir_id()))
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
            ImplItemKind::Fn(..) | ImplItemKind::TyAlias(..) => (),
        }
    }

    fn check_field_def(&mut self, cx: &LateContext<'_>, field: &hir::FieldDef<'_>) {
        let is_exported = cx.access_levels.is_exported(cx.tcx.hir().local_def_id(field.hir_id));

        self.check_ty(
            cx,
            field.ty,
            CheckTyContext {
                is_exported,
                ..CheckTyContext::default()
            },
        );
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &TraitItem<'_>) {
        let is_exported = cx.access_levels.is_exported(item.def_id);

        let context = CheckTyContext {
            is_exported,
            ..CheckTyContext::default()
        };

        match item.kind {
            TraitItemKind::Const(ty, _) | TraitItemKind::Type(_, Some(ty)) => {
                self.check_ty(cx, ty, context);
            },
            TraitItemKind::Fn(ref sig, _) => self.check_fn_decl(cx, sig.decl, context),
            TraitItemKind::Type(..) => (),
        }
    }

    fn check_local(&mut self, cx: &LateContext<'_>, local: &Local<'_>) {
        if let Some(ty) = local.ty {
            self.check_ty(
                cx,
                ty,
                CheckTyContext {
                    is_local: true,
                    ..CheckTyContext::default()
                },
            );
        }
    }
}

impl Types {
    pub fn new(vec_box_size_threshold: u64, type_complexity_threshold: u64, avoid_breaking_exported_api: bool) -> Self {
        Self {
            vec_box_size_threshold,
            type_complexity_threshold,
            avoid_breaking_exported_api,
        }
    }

    fn check_fn_decl(&mut self, cx: &LateContext<'_>, decl: &FnDecl<'_>, context: CheckTyContext) {
        // Ignore functions in trait implementations as they are usually forced by the trait definition.
        //
        // FIXME: idially we would like to warn *if the compicated type can be simplified*, but it's hard to
        // check.
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
    fn check_ty(&mut self, cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>, mut context: CheckTyContext) {
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
            TyKind::Path(ref qpath) if !context.is_local => {
                let hir_id = hir_ty.hir_id;
                let res = cx.qpath_res(qpath, hir_id);
                if let Some(def_id) = res.opt_def_id() {
                    if self.is_type_change_allowed(context) {
                        // All lints that are being checked in this block are guarded by
                        // the `avoid_breaking_exported_api` configuration. When adding a
                        // new lint, please also add the name to the configuration documentation
                        // in `clippy_lints::utils::conf.rs`

                        let mut triggered = false;
                        triggered |= box_collection::check(cx, hir_ty, qpath, def_id);
                        triggered |= redundant_allocation::check(cx, hir_ty, qpath, def_id);
                        triggered |= rc_buffer::check(cx, hir_ty, qpath, def_id);
                        triggered |= vec_box::check(cx, hir_ty, qpath, def_id, self.vec_box_size_threshold);
                        triggered |= option_option::check(cx, hir_ty, qpath, def_id);
                        triggered |= linked_list::check(cx, hir_ty, def_id);
                        triggered |= rc_mutex::check(cx, hir_ty, qpath, def_id);

                        if triggered {
                            return;
                        }
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
                            self.check_ty(cx, ty, context);
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
                            self.check_ty(cx, ty, context);
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
                                self.check_ty(cx, ty, context);
                            }
                        }
                    },
                    QPath::LangItem(..) => {},
                }
            },
            TyKind::Rptr(ref lt, ref mut_ty) => {
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

#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Default)]
struct CheckTyContext {
    is_in_trait_impl: bool,
    /// `true` for types on local variables.
    is_local: bool,
    /// `true` for types that are part of the public API.
    is_exported: bool,
    is_nested_call: bool,
}
