use crate::mir::Body;
use crate::ty::{ClosureDef, ClosureKind, FnDef, GenericArgs, IndexedVal, Ty};
use crate::{with, CrateItem, DefId, Error, ItemKind, Opaque};
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MonoItem {
    Fn(Instance),
    Static(StaticDef),
    GlobalAsm(Opaque),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Get the instance type with generic substitutions applied and lifetimes erased.
    pub fn ty(&self) -> Ty {
        with(|context| context.instance_ty(self.def))
    }

    pub fn mangled_name(&self) -> String {
        with(|context| context.instance_mangled_name(self.def))
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

/// Try to convert a crate item into an instance.
/// The item cannot be generic in order to be converted into an instance.
impl TryFrom<CrateItem> for Instance {
    type Error = crate::Error;

    fn try_from(item: CrateItem) -> Result<Self, Self::Error> {
        with(|context| {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceDef(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct StaticDef(pub DefId);

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
    pub fn ty(&self) -> Ty {
        with(|cx| cx.def_ty(self.0))
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
