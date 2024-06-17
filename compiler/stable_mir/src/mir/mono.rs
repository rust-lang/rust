use crate::abi::FnAbi;
use crate::crate_def::CrateDef;
use crate::mir::Body;
use crate::ty::{Allocation, ClosureDef, ClosureKind, FnDef, GenericArgs, IndexedVal, Ty};
use crate::{with, CrateItem, DefId, Error, ItemKind, Opaque, Symbol};
use serde::Serialize;
use std::fmt::{Debug, Formatter};
use std::io;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum MonoItem {
    Fn(Instance),
    Static(StaticDef),
    GlobalAsm(Opaque),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Instance {
    /// The type of instance.
    pub kind: InstanceKind,
    /// An ID used to get the instance definition from the compiler.
    /// Do not use this field directly.
    pub def: InstanceDef,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum InstanceKind {
    /// A user defined item.
    Item,
    /// A compiler intrinsic function.
    Intrinsic,
    /// A virtual function definition stored in a VTable.
    /// The `idx` field indicates the position in the VTable for this instance.
    Virtual { idx: usize },
    /// A compiler generated shim.
    Shim,
}

impl Instance {
    /// Get the arguments this instance was instantiated with.
    pub fn args(&self) -> GenericArgs {
        with(|cx| cx.instance_args(self.def))
    }

    /// Get the body of an Instance.
    ///
    /// The body will be eagerly monomorphized and all constants will already be evaluated.
    ///
    /// This method will return the intrinsic fallback body if one was defined.
    pub fn body(&self) -> Option<Body> {
        with(|context| context.instance_body(self.def))
    }

    /// Check whether this instance has a body available.
    ///
    /// For intrinsics with fallback body, this will return `true`. It is up to the user to decide
    /// whether to specialize the intrinsic or to use its fallback body.
    ///
    /// For more information on fallback body, see <https://github.com/rust-lang/rust/issues/93145>.
    ///
    /// This call is much cheaper than `instance.body().is_some()`, since it doesn't try to build
    /// the StableMIR body.
    pub fn has_body(&self) -> bool {
        with(|cx| cx.has_body(self.def.def_id()))
    }

    pub fn is_foreign_item(&self) -> bool {
        with(|cx| cx.is_foreign_item(self.def.def_id()))
    }

    /// Get the instance type with generic instantiations applied and lifetimes erased.
    pub fn ty(&self) -> Ty {
        with(|context| context.instance_ty(self.def))
    }

    /// Retrieve information about this instance binary interface.
    pub fn fn_abi(&self) -> Result<FnAbi, Error> {
        with(|cx| cx.instance_abi(self.def))
    }

    /// Retrieve the instance's mangled name used for calling the given instance.
    ///
    /// This will also look up the correct name of instances from upstream crates.
    pub fn mangled_name(&self) -> Symbol {
        with(|context| context.instance_mangled_name(self.def))
    }

    /// Retrieve the instance name for diagnostic messages.
    ///
    /// This will return the specialized name, e.g., `std::vec::Vec<u8>::new`.
    pub fn name(&self) -> Symbol {
        with(|context| context.instance_name(self.def, false))
    }

    /// Return a trimmed name of the given instance including its args.
    ///
    /// If a symbol name can only be imported from one place for a type, and as
    /// long as it was not glob-imported anywhere in the current crate, we trim its
    /// path and print only the name.
    pub fn trimmed_name(&self) -> Symbol {
        with(|context| context.instance_name(self.def, true))
    }

    /// Retrieve the plain intrinsic name of an instance if it's an intrinsic.
    ///
    /// The plain name does not include type arguments (as `trimmed_name` does),
    /// which is more convenient to match with intrinsic symbols.
    pub fn intrinsic_name(&self) -> Option<Symbol> {
        match self.kind {
            InstanceKind::Intrinsic => {
                Some(with(|context| context.intrinsic(self.def.def_id()).unwrap().fn_name()))
            }
            InstanceKind::Item | InstanceKind::Virtual { .. } | InstanceKind::Shim => None,
        }
    }

    /// Resolve an instance starting from a function definition and generic arguments.
    pub fn resolve(def: FnDef, args: &GenericArgs) -> Result<Instance, crate::Error> {
        with(|context| {
            context.resolve_instance(def, args).ok_or_else(|| {
                crate::Error::new(format!("Failed to resolve `{def:?}` with `{args:?}`"))
            })
        })
    }

    /// Resolve the drop in place for a given type.
    pub fn resolve_drop_in_place(ty: Ty) -> Instance {
        with(|cx| cx.resolve_drop_in_place(ty))
    }

    /// Resolve an instance for a given function pointer.
    pub fn resolve_for_fn_ptr(def: FnDef, args: &GenericArgs) -> Result<Instance, crate::Error> {
        with(|context| {
            context.resolve_for_fn_ptr(def, args).ok_or_else(|| {
                crate::Error::new(format!("Failed to resolve `{def:?}` with `{args:?}`"))
            })
        })
    }

    /// Resolve a closure with the expected kind.
    pub fn resolve_closure(
        def: ClosureDef,
        args: &GenericArgs,
        kind: ClosureKind,
    ) -> Result<Instance, crate::Error> {
        with(|context| {
            context.resolve_closure(def, args, kind).ok_or_else(|| {
                crate::Error::new(format!("Failed to resolve `{def:?}` with `{args:?}`"))
            })
        })
    }

    /// Check whether this instance is an empty shim.
    ///
    /// Allow users to check if this shim can be ignored when called directly.
    ///
    /// We have decided not to export different types of Shims to StableMIR users, however, this
    /// is a query that can be very helpful for users when processing DropGlue.
    ///
    /// When generating code for a Drop terminator, users can ignore an empty drop glue.
    /// These shims are only needed to generate a valid Drop call done via VTable.
    pub fn is_empty_shim(&self) -> bool {
        self.kind == InstanceKind::Shim
            && with(|cx| {
                cx.is_empty_drop_shim(self.def) || cx.is_empty_async_drop_ctor_shim(self.def)
            })
    }

    /// Try to constant evaluate the instance into a constant with the given type.
    ///
    /// This can be used to retrieve a constant that represents an intrinsic return such as
    /// `type_id`.
    pub fn try_const_eval(&self, const_ty: Ty) -> Result<Allocation, Error> {
        with(|cx| cx.eval_instance(self.def, const_ty))
    }

    /// Emit the body of this instance if it has one.
    pub fn emit_mir<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        if let Some(body) = self.body() { body.dump(w, &self.name()) } else { Ok(()) }
    }
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("kind", &self.kind)
            .field("def", &self.mangled_name())
            .field("args", &self.args())
            .finish()
    }
}

/// Try to convert a crate item into an instance.
/// The item cannot be generic in order to be converted into an instance.
impl TryFrom<CrateItem> for Instance {
    type Error = crate::Error;

    fn try_from(item: CrateItem) -> Result<Self, Self::Error> {
        with(|context| {
            let def_id = item.def_id();
            if !context.requires_monomorphization(def_id) {
                Ok(context.mono_instance(def_id))
            } else {
                Err(Error::new("Item requires monomorphization".to_string()))
            }
        })
    }
}

/// Try to convert an instance into a crate item.
/// Only user defined instances can be converted.
impl TryFrom<Instance> for CrateItem {
    type Error = crate::Error;

    fn try_from(value: Instance) -> Result<Self, Self::Error> {
        with(|context| {
            if value.kind == InstanceKind::Item && context.has_body(value.def.def_id()) {
                Ok(CrateItem(context.instance_def_id(value.def)))
            } else {
                Err(Error::new(format!("Item kind `{:?}` cannot be converted", value.kind)))
            }
        })
    }
}

impl From<Instance> for MonoItem {
    fn from(value: Instance) -> Self {
        MonoItem::Fn(value)
    }
}

impl From<StaticDef> for MonoItem {
    fn from(value: StaticDef) -> Self {
        MonoItem::Static(value)
    }
}

impl From<StaticDef> for CrateItem {
    fn from(value: StaticDef) -> Self {
        CrateItem(value.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct InstanceDef(usize);

impl CrateDef for InstanceDef {
    fn def_id(&self) -> DefId {
        with(|context| context.instance_def_id(*self))
    }
}

crate_def! {
    /// Holds information about a static variable definition.
    #[derive(Serialize)]
    pub StaticDef;
}

impl TryFrom<CrateItem> for StaticDef {
    type Error = crate::Error;

    fn try_from(value: CrateItem) -> Result<Self, Self::Error> {
        if matches!(value.kind(), ItemKind::Static) {
            Ok(StaticDef(value.0))
        } else {
            Err(Error::new(format!("Expected a static item, but found: {value:?}")))
        }
    }
}

impl TryFrom<Instance> for StaticDef {
    type Error = crate::Error;

    fn try_from(value: Instance) -> Result<Self, Self::Error> {
        StaticDef::try_from(CrateItem::try_from(value)?)
    }
}

impl From<StaticDef> for Instance {
    fn from(value: StaticDef) -> Self {
        // A static definition should always be convertible to an instance.
        with(|cx| cx.mono_instance(value.def_id()))
    }
}

impl StaticDef {
    /// Return the type of this static definition.
    pub fn ty(&self) -> Ty {
        with(|cx| cx.def_ty(self.0))
    }

    /// Evaluate a static's initializer, returning the allocation of the initializer's memory.
    pub fn eval_initializer(&self) -> Result<Allocation, Error> {
        with(|cx| cx.eval_static_initializer(*self))
    }
}

impl IndexedVal for InstanceDef {
    fn to_val(index: usize) -> Self {
        InstanceDef(index)
    }
    fn to_index(&self) -> usize {
        self.0
    }
}
