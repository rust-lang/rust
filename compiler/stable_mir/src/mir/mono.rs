use crate::mir::Body;
use crate::ty::{FnDef, GenericArgs, IndexedVal, Ty};
use crate::{with, CrateItem, DefId, Error, Opaque};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum MonoItem {
    Fn(Instance),
    Static(StaticDef),
    GlobalAsm(Opaque),
}

#[derive(Copy, Clone, Debug)]
pub struct Instance {
    /// The type of instance.
    pub kind: InstanceKind,
    /// An ID used to get the instance definition from the compiler.
    /// Do not use this field directly.
    pub def: InstanceDef,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    pub fn body(&self) -> Body {
        with(|context| context.instance_body(self.def))
    }

    /// Get the instance type with generic substitutions applied and lifetimes erased.
    pub fn ty(&self) -> Ty {
        with(|context| context.instance_ty(self.def))
    }

    /// Resolve an instance starting from a function definition and generic arguments.
    pub fn resolve(def: FnDef, args: &GenericArgs) -> Result<Instance, crate::Error> {
        with(|context| {
            context.resolve_instance(def, args).ok_or_else(|| {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InstanceDef(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StaticDef(pub DefId);

impl IndexedVal for InstanceDef {
    fn to_val(index: usize) -> Self {
        InstanceDef(index)
    }
    fn to_index(&self) -> usize {
        self.0
    }
}
