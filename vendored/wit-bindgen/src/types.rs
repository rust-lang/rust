use std::collections::HashMap;
use wit_parser::*;

#[derive(Default)]
pub struct Types {
    type_info: HashMap<TypeId, TypeInfo>,
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct TypeInfo {
    /// Whether or not this type is ever used (transitively) within a borrowed
    /// context, or a parameter to an export function.
    pub borrowed: bool,

    /// Whether or not this type is ever used (transitively) within an owned
    /// context, such as the result of an exported function or in the params or
    /// results of an imported function.
    pub owned: bool,

    /// Whether or not this type is ever used (transitively) within the
    /// error case in the result of a function.
    pub error: bool,

    /// Whether or not this type (transitively) has a list.
    pub has_list: bool,

    /// Whether or not this type (transitively) has a handle.
    pub has_handle: bool,

    /// Whether or not this type (transitively) has a future or a stream.
    pub has_future_or_stream: bool,
}

impl std::ops::BitOrAssign for TypeInfo {
    fn bitor_assign(&mut self, rhs: Self) {
        self.borrowed |= rhs.borrowed;
        self.owned |= rhs.owned;
        self.error |= rhs.error;
        self.has_list |= rhs.has_list;
        self.has_handle |= rhs.has_handle;
        self.has_future_or_stream |= rhs.has_future_or_stream;
    }
}

impl Types {
    pub fn analyze(&mut self, resolve: &Resolve, world: WorldId) {
        // Build up all type information first which is inherited through types,
        // such as properties of borrows/lists/etc.
        for (t, _) in resolve.types.iter() {
            self.type_id_info(resolve, t);
        }

        // ... next handle borrowed/owned flags which aren't inherited through
        // types.
        let world = &resolve.worlds[world];
        for (import, (_, item)) in world
            .imports
            .iter()
            .map(|i| (true, i))
            .chain(world.exports.iter().map(|i| (false, i)))
        {
            match item {
                WorldItem::Function(f) => self.type_info_func(resolve, f, import),
                WorldItem::Interface { id, .. } => {
                    let iface = &resolve.interfaces[*id];

                    for (_, t) in iface.types.iter() {
                        self.type_id_info(resolve, *t);
                    }
                    for (_, f) in iface.functions.iter() {
                        self.type_info_func(resolve, f, import);
                    }
                }
                WorldItem::Type { id, .. } => {
                    self.type_id_info(resolve, *id);
                }
            }
        }
    }

    fn type_info_func(&mut self, resolve: &Resolve, func: &Function, import: bool) {
        let mut live = LiveTypes::default();
        for param in func.params.iter() {
            self.type_info(resolve, &param.ty);
            live.add_type(resolve, &param.ty);
        }
        for id in live.iter() {
            if resolve.types[id].name.is_some() {
                let info = self.type_info.get_mut(&id).unwrap();
                if import {
                    info.owned = true;
                } else {
                    info.borrowed = true;
                }
            }
        }
        let mut live = LiveTypes::default();
        if let Some(ty) = &func.result {
            self.type_info(resolve, ty);
            live.add_type(resolve, ty);
        }
        for id in live.iter() {
            if resolve.types[id].name.is_some() {
                self.type_info.get_mut(&id).unwrap().owned = true;
            }
        }

        if let Some(Type::Id(id)) = func.result {
            if let TypeDefKind::Result(Result_ {
                err: Some(Type::Id(id)),
                ..
            }) = &resolve.types[id].kind
            {
                let id = super::resolve_type_definition_id(resolve, *id);
                self.type_info.get_mut(&id).unwrap().error = true;
            }
        }
    }

    pub fn get(&self, id: TypeId) -> TypeInfo {
        self.type_info[&id]
    }

    fn type_id_info(&mut self, resolve: &Resolve, ty: TypeId) -> TypeInfo {
        if let Some(info) = self.type_info.get(&ty) {
            return *info;
        }
        let mut info = TypeInfo::default();
        match &resolve.types[ty].kind {
            TypeDefKind::Record(r) => {
                for field in r.fields.iter() {
                    info |= self.type_info(resolve, &field.ty);
                }
            }
            TypeDefKind::Tuple(t) => {
                for ty in t.types.iter() {
                    info |= self.type_info(resolve, ty);
                }
            }
            TypeDefKind::Flags(_) => {}
            TypeDefKind::Enum(_) => {}
            TypeDefKind::Variant(v) => {
                for case in v.cases.iter() {
                    info |= self.optional_type_info(resolve, case.ty.as_ref());
                }
            }
            TypeDefKind::List(ty) => {
                info = self.type_info(resolve, ty);
                info.has_list = true;
            }
            TypeDefKind::Map(k, v) => {
                info = self.type_info(resolve, k);
                info |= self.type_info(resolve, v);
                info.has_list = true;
            }
            TypeDefKind::Type(ty) | TypeDefKind::Option(ty) => {
                info = self.type_info(resolve, ty);
            }
            TypeDefKind::Result(r) => {
                info = self.optional_type_info(resolve, r.ok.as_ref());
                info |= self.optional_type_info(resolve, r.err.as_ref());
            }
            TypeDefKind::Future(ty) | TypeDefKind::Stream(ty) => {
                info = self.optional_type_info(resolve, ty.as_ref());
                info.has_future_or_stream = true;
            }
            TypeDefKind::Handle(_) => info.has_handle = true,
            TypeDefKind::Resource => {}
            TypeDefKind::Unknown => unreachable!(),
            TypeDefKind::FixedLengthList(..) => todo!(),
        }
        self.type_info.insert(ty, info);
        info
    }

    fn type_info(&mut self, resolve: &Resolve, ty: &Type) -> TypeInfo {
        let mut info = TypeInfo::default();
        match ty {
            Type::String => info.has_list = true,
            Type::Id(id) => return self.type_id_info(resolve, *id),
            _ => {}
        }
        info
    }

    fn optional_type_info(&mut self, resolve: &Resolve, ty: Option<&Type>) -> TypeInfo {
        match ty {
            Some(ty) => self.type_info(resolve, ty),
            None => TypeInfo::default(),
        }
    }
}

impl TypeInfo {
    pub fn is_copy(&self) -> bool {
        !self.has_list && !self.has_handle && !self.has_future_or_stream
    }

    pub fn is_clone(&self) -> bool {
        !self.has_handle && !self.has_future_or_stream
    }
}
