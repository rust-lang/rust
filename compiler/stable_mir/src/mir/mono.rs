use crate::crate_def::CrateDef;
use crate::mir::Body;
use crate::ty::{Allocation, ClosureDef, ClosureKind, FnDef, GenericArgs, IndexedVal, Ty};
use crate::{with, CrateItem, DefId, Error, ItemKind, Opaque, Symbol};
use std::fmt::{Debug, Formatter};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MonoItem {
    Fn(Instance),
    Static(StaticDef),
    GlobalAsm(Opaque),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Instance {
    /// The type of instance.
    pub kind: InstanceKind,
    /// An ID used to get the instance definition from the compiler.
    /// Do not use this field directly.
    pub def: InstanceDef,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstanceKind {
    /// A user defined item.
    Item,
    /// A compiler intrinsic function.
    Intrinsic,
    /// A virtual function definition stored in a VTable.
    Virtual,
    /// A compiler generated shim.
    Shim,
}

impl Instance {
    /// Get the body of an Instance. The body will be eagerly monomorphized.
    pub fn body(&self) -> Option<Body> {
        with(|context| context.instance_body(self.def))
    }

    pub fn is_foreign_item(&self) -> bool {
        let item = CrateItem::try_from(*self);
        item.as_ref().map_or(false, CrateItem::is_foreign_item)
    }

    /// Get the instance type with generic substitutions applied and lifetimes erased.
    pub fn ty(&self) -> Ty {
        with(|context| context.instance_ty(self.def))
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
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("kind", &self.kind)
            .field("def", &self.mangled_name())
            .finish()
    }
}

/// Try to convert a crate item into an instance.
/// The item cannot be generic in order to be converted into an instance.
impl TryFrom<CrateItem> for Instance {
    type Error = crate::Error;

    fn try_from(item: CrateItem) -> Result<Self, Self::Error> {
        with(|context| {
            // FIXME(celinval):
            // - Return `Err` if instance does not have a body.
            if !context.requires_monomorphization(item.0) {
                Ok(context.mono_instance(item))
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
        if value.kind == InstanceKind::Item {
            Ok(CrateItem(with(|context| context.instance_def_id(value.def))))
        } else {
            Err(Error::new(format!("Item kind `{:?}` cannot be converted", value.kind)))
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceDef(usize);

crate_def! {
    /// Holds information about a static variable definition.
    pub StaticDef;
}

impl TryFrom<CrateItem> for StaticDef {
    type Error = crate::Error;

    fn try_from(value: CrateItem) -> Result<Self, Self::Error> {
        if matches!(value.kind(), ItemKind::Static | ItemKind::Const) {
            Ok(StaticDef(value.0))
        } else {
            Err(Error::new(format!("Expected a static item, but found: {value:?}")))
        }
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
