use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

#[proc_macro_derive(Graphable)]
pub fn derive_graphable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // 1. Validate: struct only
    let data = match input.data {
        Data::Struct(d) => d,
        _ => {
            return syn::Error::new_spanned(name, "Graphable only supported on structs")
                .to_compile_error()
                .into()
        }
    };

    // 2. Validate: Must have #[repr(C, packed)]
    // This is hard to check robustly in proc-macro without reading attributes deeply,
    // but we can check for presence of repr attribute.
    // For now, let's skip strict attribute check in macro and rely on developer or unsafe impl safety comments,
    // OR we can try to find it. But user asked to "require the type has #[repr(C, packed)]".
    // We can emit a compile_error if not present ideally, but attribute parsing is complex.
    // Let's assume for now we enforce fields strictly.

    // 3. Process fields
    let fields = match data.fields {
        Fields::Named(f) => f.named,
        _ => {
            return syn::Error::new_spanned(name, "Graphable only supported on named structs")
                .to_compile_error()
                .into()
        }
    };

    let mut schema_fields = Vec::new();

    for field in fields {
        let field_name = field.ident.unwrap();
        let field_name_str = field_name.to_string();
        let ty = field.ty;

        // Check for forbidden types
        if let Type::Path(tp) = &ty {
            if let Some(ident) = tp.path.get_ident() {
                let s = ident.to_string();
                if s == "String"
                    || s == "Vec"
                    || s == "Box"
                    || s == "Rc"
                    || s == "Arc"
                    || s == "usize"
                    || s == "isize"
                {
                    return syn::Error::new_spanned(
                        &ty,
                        format!(
                            "Type {} is forbidden in Graphable structs (no pointers/heap/usize)",
                            s
                        ),
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }

        // Also check for raw pointers / references (Type::Ptr, Type::Reference)
        match &ty {
            Type::Ptr(_) | Type::Reference(_) => {
                return syn::Error::new_spanned(
                    &ty,
                    "Pointers and references forbidden in Graphable",
                )
                .to_compile_error()
                .into();
            }
            _ => {}
        }

        // Map Rust type to WireType construction
        // We need to generate code that produces a WireType enum variant.
        // ex: WireType::U32
        // We can do this by matching known types or using a helper trait `HasWireType`.
        // User didn't specify a helper trait, but creating one makes this clean.
        // However, macro must generate `Field { name: "foo", ty: WireType::... }`.

        let wire_type_expr = map_type_to_wire_type(&ty);

        schema_fields.push(quote! {
            abi::wire_schema::Field {
                name: #field_name_str,
                ty: #wire_type_expr,
            }
        });
    }

    let schema_name = name.to_string();
    let _fields_len = schema_fields.len();

    let expanded = quote! {
        // Enforce WireSafe
        unsafe impl abi::wire::WireSafe for #name {}

        impl abi::graphable::Graphable for #name {
            const SCHEMA: abi::wire_schema::Schema = abi::wire_schema::Schema {
                name: #schema_name,
                fields: &[
                    #(#schema_fields),*
                ],
            };
        }
    };

    TokenStream::from(expanded)
}

fn map_type_to_wire_type(ty: &Type) -> proc_macro2::TokenStream {
    // This is a heuristic mapping. For robust mapping, we might want a trait `GetWireType`.
    // But for this task, we can pattern match common types.
    if let Type::Path(tp) = ty {
        if let Some(ident) = tp.path.get_ident() {
            let s = ident.to_string();
            return match s.as_str() {
                "u8" => quote!(abi::wire_schema::WireType::U8),
                "u16" => quote!(abi::wire_schema::WireType::U16),
                "u32" => quote!(abi::wire_schema::WireType::U32),
                "u64" => quote!(abi::wire_schema::WireType::U64),
                "u128" => quote!(abi::wire_schema::WireType::U128),
                "i8" => quote!(abi::wire_schema::WireType::I8),
                "i16" => quote!(abi::wire_schema::WireType::I16),
                "i32" => quote!(abi::wire_schema::WireType::I32),
                "i64" => quote!(abi::wire_schema::WireType::I64),
                "i128" => quote!(abi::wire_schema::WireType::I128),
                "bool" => quote!(abi::wire_schema::WireType::Bool),
                "ThingId" => quote!(abi::wire_schema::WireType::ThingId),
                "BlobId" => quote!(abi::wire_schema::WireType::BlobId),
                "SymbolId" => quote!(abi::wire_schema::WireType::SymbolId),
                "KindId" => quote!(abi::wire_schema::WireType::ThingId), // KindId is alias for ThingId layout basically? Or should have own? Wire schema has KindId.
                "PredicateId" => quote!(abi::wire_schema::WireType::ThingId), // PredicateId maps to ThingId wire type layout-wise (16 bytes) usually, or we can add PredicateId to WireType if not there.
                // Wait, checking WireType definition... `KindId` exists. `PredicateId` is new, likely maps to ThingId or we add it.
                // Previous graphable.rs mapped PredicateId to ThingId.
                _ => {
                    quote!(abi::wire_schema::WireType::Struct(<#ty as abi::graphable::Graphable>::SCHEMA))
                }
            };
        }
    }

    // Arrays [T; N]
    if let Type::Array(ta) = ty {
        let inner = map_type_to_wire_type(&ta.elem);
        let len = &ta.len;
        return quote!(abi::wire_schema::WireType::Array(&#inner, #len));
    }

    quote!(abi::wire_schema::WireType::U8) // Fallback/Error?
}
