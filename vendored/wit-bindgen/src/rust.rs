use crate::{Ownership, types::TypeInfo};
use heck::*;
use wit_parser::*;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TypeMode {
    Owned,
    AllBorrowed(&'static str),
}

pub trait RustGenerator<'a> {
    fn resolve(&self) -> &'a Resolve;

    fn push_str(&mut self, s: &str);
    fn info(&self, ty: TypeId) -> TypeInfo;
    fn path_to_interface(&self, interface: InterfaceId) -> Option<String>;
    fn is_imported_interface(&self, interface: InterfaceId) -> bool;
    fn wasmtime_path(&self) -> String;

    /// This determines whether we generate owning types or (where appropriate)
    /// borrowing types.
    ///
    /// For example, when generating a type which is only used as a parameter to
    /// a guest-exported function, there is no need for it to own its fields.
    /// However, constructing deeply-nested borrows (e.g. `&[&[&[&str]]]]` for
    /// `list<list<list<string>>>`) can be very awkward, so by default we
    /// generate owning types and use only shallow borrowing at the top level
    /// inside function signatures.
    fn ownership(&self) -> Ownership;

    fn print_ty(&mut self, ty: &Type, mode: TypeMode) {
        self.push_str(&self.ty(ty, mode))
    }
    fn ty(&self, ty: &Type, mode: TypeMode) -> String {
        match ty {
            Type::Id(t) => self.tyid(*t, mode),
            Type::Bool => "bool".to_string(),
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::S8 => "i8".to_string(),
            Type::S16 => "i16".to_string(),
            Type::S32 => "i32".to_string(),
            Type::S64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::Char => "char".to_string(),
            Type::String => match mode {
                TypeMode::AllBorrowed(lt) => {
                    if lt != "'_" {
                        format!("&{lt} str")
                    } else {
                        format!("&str")
                    }
                }
                TypeMode::Owned => {
                    let wt = self.wasmtime_path();
                    format!("{wt}::component::__internal::String")
                }
            },
            Type::ErrorContext => {
                let wt = self.wasmtime_path();
                format!("{wt}::component::ErrorContext")
            }
        }
    }

    fn print_optional_ty(&mut self, ty: Option<&Type>, mode: TypeMode) {
        self.push_str(&self.optional_ty(ty, mode))
    }
    fn optional_ty(&self, ty: Option<&Type>, mode: TypeMode) -> String {
        match ty {
            Some(ty) => self.ty(ty, mode),
            None => "()".to_string(),
        }
    }

    fn tyid(&self, id: TypeId, mode: TypeMode) -> String {
        let info = self.info(id);
        let lt = self.lifetime_for(&info, mode);
        let ty = &self.resolve().types[id];
        if ty.name.is_some() {
            // If this type has a list internally, no lifetime is being printed,
            // but we're in a borrowed mode, then that means we're in a borrowed
            // context and don't want ownership of the type but we're using an
            // owned type definition. Inject a `&` in front to indicate that, at
            // the API level, ownership isn't required.
            let mut out = String::new();
            if info.has_list && lt.is_none() {
                if let TypeMode::AllBorrowed(lt) = mode {
                    if lt != "'_" {
                        out.push_str(&format!("&{lt} "))
                    } else {
                        out.push_str("&")
                    }
                }
            }
            let name = if lt.is_some() {
                self.param_name(id)
            } else {
                self.result_name(id)
            };
            out.push_str(&self.type_name_in_interface(ty.owner, &name));

            // If the type recursively owns data and it's a
            // variant/record/list, then we need to place the
            // lifetime parameter on the type as well.
            if info.has_list && needs_generics(self.resolve(), &ty.kind) {
                out.push_str(&self.generics(lt));
            }

            return out;

            fn needs_generics(resolve: &Resolve, ty: &TypeDefKind) -> bool {
                match ty {
                    TypeDefKind::Variant(_)
                    | TypeDefKind::Record(_)
                    | TypeDefKind::Option(_)
                    | TypeDefKind::Result(_)
                    | TypeDefKind::Future(_)
                    | TypeDefKind::Stream(_)
                    | TypeDefKind::List(_)
                    | TypeDefKind::Map(_, _)
                    | TypeDefKind::Flags(_)
                    | TypeDefKind::Enum(_)
                    | TypeDefKind::Tuple(_)
                    | TypeDefKind::Handle(_)
                    | TypeDefKind::Resource => true,
                    TypeDefKind::Type(Type::Id(t)) => {
                        needs_generics(resolve, &resolve.types[*t].kind)
                    }
                    TypeDefKind::Type(Type::String) => true,
                    TypeDefKind::Type(_) => false,
                    TypeDefKind::Unknown => unreachable!(),
                    TypeDefKind::FixedLengthList(..) => todo!(),
                }
            }
        }

        match &ty.kind {
            TypeDefKind::List(t) => self.list(t, mode),

            TypeDefKind::Option(t) => {
                format!("Option<{}>", self.ty(t, mode))
            }

            TypeDefKind::Result(r) => {
                let ok = self.optional_ty(r.ok.as_ref(), mode);
                let err = self.optional_ty(r.err.as_ref(), mode);
                format!("Result<{ok},{err}>")
            }

            TypeDefKind::Variant(_) => panic!("unsupported anonymous variant"),

            // Tuple-like records are mapped directly to Rust tuples of
            // types. Note the trailing comma after each member to
            // appropriately handle 1-tuples.
            TypeDefKind::Tuple(t) => {
                let mut out = "(".to_string();
                for ty in t.types.iter() {
                    out.push_str(&self.ty(ty, mode));
                    out.push_str(",");
                }
                out.push_str(")");
                out
            }
            TypeDefKind::Record(_) => {
                panic!("unsupported anonymous type reference: record")
            }
            TypeDefKind::Flags(_) => {
                panic!("unsupported anonymous type reference: flags")
            }
            TypeDefKind::Enum(_) => {
                panic!("unsupported anonymous type reference: enum")
            }
            TypeDefKind::Future(ty) => {
                let wt = self.wasmtime_path();
                let t = self.optional_ty(ty.as_ref(), TypeMode::Owned);
                format!("{wt}::component::FutureReader<{t}>")
            }
            TypeDefKind::Stream(ty) => {
                let wt = self.wasmtime_path();
                let t = self.optional_ty(ty.as_ref(), TypeMode::Owned);
                format!("{wt}::component::StreamReader<{t}>")
            }
            TypeDefKind::Handle(handle) => self.handle(handle),
            TypeDefKind::Resource => unreachable!(),

            TypeDefKind::Type(t) => self.ty(t, mode),
            TypeDefKind::Map(k, v) => {
                let key = self.ty(k, mode);
                let value = self.ty(v, mode);
                format!("std::collections::HashMap<{key}, {value}>")
            }
            TypeDefKind::Unknown => unreachable!(),
            TypeDefKind::FixedLengthList(..) => todo!(),
        }
    }

    fn type_name_in_interface(&self, owner: TypeOwner, name: &str) -> String {
        let mut out = String::new();
        if let TypeOwner::Interface(id) = owner {
            if let Some(path) = self.path_to_interface(id) {
                out.push_str(&path);
                out.push_str("::");
            }
        }
        out.push_str(name);
        out
    }

    fn print_list(&mut self, ty: &Type, mode: TypeMode) {
        self.push_str(&self.list(ty, mode))
    }
    fn list(&self, ty: &Type, mode: TypeMode) -> String {
        let next_mode = if matches!(self.ownership(), Ownership::Owning) {
            TypeMode::Owned
        } else {
            mode
        };
        let ty = self.ty(ty, next_mode);
        match mode {
            TypeMode::AllBorrowed(lt) => {
                if lt != "'_" {
                    format!("&{lt} [{ty}]")
                } else {
                    format!("&[{ty}]")
                }
            }
            TypeMode::Owned => {
                let wt = self.wasmtime_path();
                format!("{wt}::component::__internal::Vec<{ty}>")
            }
        }
    }

    fn print_stream(&mut self, ty: Option<&Type>) {
        self.push_str(&self.stream(ty))
    }
    fn stream(&self, ty: Option<&Type>) -> String {
        let wt = self.wasmtime_path();
        let mut out = format!("{wt}::component::HostStream<");
        out.push_str(&self.optional_ty(ty, TypeMode::Owned));
        out.push_str(">");
        out
    }

    fn print_future(&mut self, ty: Option<&Type>) {
        self.push_str(&self.future(ty))
    }
    fn future(&self, ty: Option<&Type>) -> String {
        let wt = self.wasmtime_path();
        let mut out = format!("{wt}::component::HostFuture<");
        out.push_str(&self.optional_ty(ty, TypeMode::Owned));
        out.push_str(">");
        out
    }

    fn print_handle(&mut self, handle: &Handle) {
        self.push_str(&self.handle(handle))
    }
    fn handle(&self, handle: &Handle) -> String {
        // Handles are either printed as `ResourceAny` for any guest-defined
        // resource or `Resource<T>` for all host-defined resources. This means
        // that this function needs to determine if `handle` points to a host
        // or a guest resource which is determined by:
        //
        // * For world-owned resources, they're always imported.
        // * For interface-owned resources, it depends on the how bindings were
        //   last generated for this interface.
        //
        // Additionally type aliases via `use` are "peeled" here to find the
        // original definition of the resource since that's the one that we
        // care about for determining whether it's imported or not.
        let resource = match handle {
            Handle::Own(t) | Handle::Borrow(t) => *t,
        };
        let ty = &self.resolve().types[resource];
        let def_id = super::resolve_type_definition_id(self.resolve(), resource);
        let ty_def = &self.resolve().types[def_id];
        let is_host_defined = match ty_def.owner {
            TypeOwner::Interface(i) => self.is_imported_interface(i),
            _ => true,
        };
        let wt = self.wasmtime_path();
        if is_host_defined {
            let mut out = format!("{wt}::component::Resource<");
            out.push_str(&self.type_name_in_interface(
                ty.owner,
                &ty.name.as_ref().unwrap().to_upper_camel_case(),
            ));
            out.push_str(">");
            out
        } else {
            format!("{wt}::component::ResourceAny")
        }
    }

    fn print_generics(&mut self, lifetime: Option<&str>) {
        self.push_str(&self.generics(lifetime))
    }
    fn generics(&self, lifetime: Option<&str>) -> String {
        if let Some(lt) = lifetime {
            format!("<{lt},>")
        } else {
            String::new()
        }
    }

    fn modes_of(&self, ty: TypeId) -> Vec<(String, TypeMode)> {
        let info = self.info(ty);
        // Info only populated for types that are passed to and from functions. For
        // types which are not, default to the ownership setting.
        if !info.owned && !info.borrowed {
            return vec![(
                self.param_name(ty),
                match self.ownership() {
                    Ownership::Owning => TypeMode::Owned,
                    Ownership::Borrowing { .. } => TypeMode::AllBorrowed("'a"),
                },
            )];
        }
        let mut result = Vec::new();
        let first_mode =
            if info.owned || !info.borrowed || matches!(self.ownership(), Ownership::Owning) {
                TypeMode::Owned
            } else {
                assert!(!self.uses_two_names(&info));
                TypeMode::AllBorrowed("'a")
            };
        result.push((self.result_name(ty), first_mode));
        if self.uses_two_names(&info) {
            result.push((self.param_name(ty), TypeMode::AllBorrowed("'a")));
        }
        result
    }

    fn param_name(&self, ty: TypeId) -> String {
        let info = self.info(ty);
        let name = self.resolve().types[ty]
            .name
            .as_ref()
            .unwrap()
            .to_upper_camel_case();
        if self.uses_two_names(&info) {
            format!("{name}Param")
        } else {
            name
        }
    }

    fn result_name(&self, ty: TypeId) -> String {
        let info = self.info(ty);
        let name = self.resolve().types[ty]
            .name
            .as_ref()
            .unwrap()
            .to_upper_camel_case();
        if self.uses_two_names(&info) {
            format!("{name}Result")
        } else {
            name
        }
    }

    fn uses_two_names(&self, info: &TypeInfo) -> bool {
        info.has_list
            && info.borrowed
            && info.owned
            && matches!(
                self.ownership(),
                Ownership::Borrowing {
                    duplicate_if_necessary: true
                }
            )
    }

    fn lifetime_for(&self, info: &TypeInfo, mode: TypeMode) -> Option<&'static str> {
        if matches!(self.ownership(), Ownership::Owning) {
            return None;
        }
        let lt = match mode {
            TypeMode::AllBorrowed(s) => s,
            _ => return None,
        };
        // No lifetimes needed unless this has a list.
        if !info.has_list {
            return None;
        }
        // If two names are used then this type will have an owned and a
        // borrowed copy and the borrowed copy is being used, so it needs a
        // lifetime. Otherwise if it's only borrowed and not owned then this can
        // also use a lifetime since it's not needed in two contexts and only
        // the borrowed version of the structure was generated.
        if self.uses_two_names(info) || (info.borrowed && !info.owned) {
            Some(lt)
        } else {
            None
        }
    }

    fn typedfunc_sig(&self, func: &Function, param_mode: TypeMode) -> String {
        let mut out = "(".to_string();
        for param in func.params.iter() {
            out.push_str(&self.ty(&param.ty, param_mode));
            out.push_str(", ");
        }
        out.push_str("), (");
        if let Some(ty) = func.result {
            out.push_str(&self.ty(&ty, TypeMode::Owned));
            out.push_str(", ");
        }
        out.push_str(")");
        out
    }
}

/// Translate `name` to a Rust `snake_case` identifier.
pub fn to_rust_ident(name: &str) -> String {
    match name {
        // Escape Rust keywords.
        // Source: https://doc.rust-lang.org/reference/keywords.html
        "as" => "as_".into(),
        "break" => "break_".into(),
        "const" => "const_".into(),
        "continue" => "continue_".into(),
        "crate" => "crate_".into(),
        "else" => "else_".into(),
        "enum" => "enum_".into(),
        "extern" => "extern_".into(),
        "false" => "false_".into(),
        "fn" => "fn_".into(),
        "for" => "for_".into(),
        "if" => "if_".into(),
        "impl" => "impl_".into(),
        "in" => "in_".into(),
        "let" => "let_".into(),
        "loop" => "loop_".into(),
        "match" => "match_".into(),
        "mod" => "mod_".into(),
        "move" => "move_".into(),
        "mut" => "mut_".into(),
        "pub" => "pub_".into(),
        "ref" => "ref_".into(),
        "return" => "return_".into(),
        "self" => "self_".into(),
        "static" => "static_".into(),
        "struct" => "struct_".into(),
        "super" => "super_".into(),
        "trait" => "trait_".into(),
        "true" => "true_".into(),
        "type" => "type_".into(),
        "unsafe" => "unsafe_".into(),
        "use" => "use_".into(),
        "where" => "where_".into(),
        "while" => "while_".into(),
        "async" => "async_".into(),
        "await" => "await_".into(),
        "dyn" => "dyn_".into(),
        "abstract" => "abstract_".into(),
        "become" => "become_".into(),
        "box" => "box_".into(),
        "do" => "do_".into(),
        "final" => "final_".into(),
        "macro" => "macro_".into(),
        "override" => "override_".into(),
        "priv" => "priv_".into(),
        "typeof" => "typeof_".into(),
        "unsized" => "unsized_".into(),
        "virtual" => "virtual_".into(),
        "yield" => "yield_".into(),
        "try" => "try_".into(),
        "gen" => "gen_".into(),
        s => s.to_snake_case(),
    }
}

/// Translate `name` to a Rust `UpperCamelCase` identifier.
pub fn to_rust_upper_camel_case(name: &str) -> String {
    match name {
        // We use `Host` as the name of the trait for host implementations
        // to fill in, so rename it if "Host" is used as a regular identifier.
        "host" => "Host_".into(),
        s => s.to_upper_camel_case(),
    }
}
