#![deny(rust_2018_idioms)]
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro::TokenStream;
use std::{fs::File, io::Read, path::Path};
use syn::ext::IdentExt;
use syn::parse::Parser as _;

#[proc_macro]
pub fn x86_functions(input: TokenStream) -> TokenStream {
    functions(input, &["core_arch/src/x86", "core_arch/src/x86_64"])
}

#[proc_macro]
pub fn arm_functions(input: TokenStream) -> TokenStream {
    functions(
        input,
        &[
            "core_arch/src/arm",
            "core_arch/src/aarch64",
            "core_arch/src/arm_shared/neon",
        ],
    )
}

#[proc_macro]
pub fn mips_functions(input: TokenStream) -> TokenStream {
    functions(input, &["core_arch/src/mips"])
}

fn functions(input: TokenStream, dirs: &[&str]) -> TokenStream {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = dir.parent().expect("root-dir not found");

    let mut files = Vec::new();
    for dir in dirs {
        walk(&root.join(dir), &mut files);
    }
    assert!(!files.is_empty());

    let mut functions = Vec::new();
    for &mut (ref mut file, ref path) in &mut files {
        for mut item in file.items.drain(..) {
            match item {
                syn::Item::Fn(f) => functions.push((f, path)),
                syn::Item::Mod(ref mut m) => {
                    if let Some(ref mut m) = m.content {
                        for i in m.1.drain(..) {
                            if let syn::Item::Fn(f) = i {
                                functions.push((f, path))
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }
    assert!(!functions.is_empty());

    let mut tests = std::collections::HashSet::<String>::new();
    for f in &functions {
        let id = format!("{}", f.0.sig.ident);
        if id.starts_with("test_") {
            tests.insert(id);
        }
    }
    assert!(!tests.is_empty());

    functions.retain(|(f, _)| matches!(f.vis, syn::Visibility::Public(_)));
    assert!(!functions.is_empty());

    let input = proc_macro2::TokenStream::from(input);

    let functions = functions
        .iter()
        .map(|&(ref f, path)| {
            let name = &f.sig.ident;
            // println!("{name}");
            let mut arguments = Vec::new();
            let mut const_arguments = Vec::new();
            for input in f.sig.inputs.iter() {
                let ty = match *input {
                    syn::FnArg::Typed(ref c) => &c.ty,
                    _ => panic!("invalid argument on {name}"),
                };
                arguments.push(to_type(ty));
            }
            for generic in f.sig.generics.params.iter() {
                match *generic {
                    syn::GenericParam::Const(ref c) => const_arguments.push(to_type(&c.ty)),
                    syn::GenericParam::Type(ref _t) => (),
                    _ => panic!("invalid generic argument on {name}"),
                };
            }
            let ret = match f.sig.output {
                syn::ReturnType::Default => quote! { None },
                syn::ReturnType::Type(_, ref t) => {
                    let ty = to_type(t);
                    quote! { Some(#ty) }
                }
            };
            let instrs = find_instrs(&f.attrs);
            let target_feature = if let Some(i) = find_target_feature(&f.attrs) {
                quote! { Some(#i) }
            } else {
                quote! { None }
            };

            let required_const = find_required_const("rustc_args_required_const", &f.attrs);
            let mut legacy_const_generics =
                find_required_const("rustc_legacy_const_generics", &f.attrs);
            if !required_const.is_empty() && !legacy_const_generics.is_empty() {
                panic!(
                    "Can't have both #[rustc_args_required_const] and \
                     #[rustc_legacy_const_generics]"
                );
            }

            // The list of required consts, used to verify the arguments, comes from either the
            // `rustc_args_required_const` or the `rustc_legacy_const_generics` attribute.
            let required_const = if required_const.is_empty() {
                legacy_const_generics.clone()
            } else {
                required_const
            };

            legacy_const_generics.sort();
            for (idx, ty) in legacy_const_generics
                .into_iter()
                .zip(const_arguments.into_iter())
            {
                arguments.insert(idx, ty);
            }

            // strip leading underscore from fn name when building a test
            // _mm_foo -> mm_foo such that the test name is test_mm_foo.
            let test_name_string = format!("{name}");
            let mut test_name_id = test_name_string.as_str();
            while test_name_id.starts_with('_') {
                test_name_id = &test_name_id[1..];
            }
            let has_test = tests.contains(&format!("test_{test_name_id}"));

            let doc = find_doc(&f.attrs);

            quote! {
                Function {
                    name: stringify!(#name),
                    arguments: &[#(#arguments),*],
                    ret: #ret,
                    target_feature: #target_feature,
                    instrs: &[#(#instrs),*],
                    file: stringify!(#path),
                    required_const: &[#(#required_const),*],
                    has_test: #has_test,
                    doc: #doc
                }
            }
        })
        .collect::<Vec<_>>();

    let ret = quote! { #input: &[Function] = &[#(#functions),*]; };
    // println!("{ret}");
    ret.into()
}

fn to_type(t: &syn::Type) -> proc_macro2::TokenStream {
    match *t {
        syn::Type::Path(ref p) => match extract_path_ident(&p.path).to_string().as_ref() {
            // x86 ...
            "__m128" => quote! { &M128 },
            "__m128bh" => quote! { &M128BH },
            "__m128d" => quote! { &M128D },
            "__m128h" => quote! { &M128H },
            "__m128i" => quote! { &M128I },
            "__m256" => quote! { &M256 },
            "__m256bh" => quote! { &M256BH },
            "__m256d" => quote! { &M256D },
            "__m256h" => quote! { &M256H },
            "__m256i" => quote! { &M256I },
            "__m512" => quote! { &M512 },
            "__m512bh" => quote! { &M512BH },
            "__m512d" => quote! { &M512D },
            "__m512h" => quote! { &M512H },
            "__m512i" => quote! { &M512I },
            "__mmask8" => quote! { &MMASK8 },
            "__mmask16" => quote! { &MMASK16 },
            "__mmask32" => quote! { &MMASK32 },
            "__mmask64" => quote! { &MMASK64 },
            "_MM_CMPINT_ENUM" => quote! { &MM_CMPINT_ENUM },
            "_MM_MANTISSA_NORM_ENUM" => quote! { &MM_MANTISSA_NORM_ENUM },
            "_MM_MANTISSA_SIGN_ENUM" => quote! { &MM_MANTISSA_SIGN_ENUM },
            "_MM_PERM_ENUM" => quote! { &MM_PERM_ENUM },
            "bool" => quote! { &BOOL },
            "bf16" => quote! { &BF16 },
            "f16" => quote! { &F16 },
            "f32" => quote! { &F32 },
            "f64" => quote! { &F64 },
            "i16" => quote! { &I16 },
            "i32" => quote! { &I32 },
            "i64" => quote! { &I64 },
            "i8" => quote! { &I8 },
            "u16" => quote! { &U16 },
            "u32" => quote! { &U32 },
            "u64" => quote! { &U64 },
            "u128" => quote! { &U128 },
            "usize" => quote! { &USIZE },
            "u8" => quote! { &U8 },
            "p8" => quote! { &P8 },
            "p16" => quote! { &P16 },
            "Ordering" => quote! { &ORDERING },
            "CpuidResult" => quote! { &CPUID },

            // arm ...
            "int8x4_t" => quote! { &I8X4 },
            "int8x8_t" => quote! { &I8X8 },
            "int8x8x2_t" => quote! { &I8X8X2 },
            "int8x8x3_t" => quote! { &I8X8X3 },
            "int8x8x4_t" => quote! { &I8X8X4 },
            "int8x16x2_t" => quote! { &I8X16X2 },
            "int8x16x3_t" => quote! { &I8X16X3 },
            "int8x16x4_t" => quote! { &I8X16X4 },
            "int8x16_t" => quote! { &I8X16 },
            "int16x2_t" => quote! { &I16X2 },
            "int16x4_t" => quote! { &I16X4 },
            "int16x4x2_t" => quote! { &I16X4X2 },
            "int16x4x3_t" => quote! { &I16X4X3 },
            "int16x4x4_t" => quote! { &I16X4X4 },
            "int16x8_t" => quote! { &I16X8 },
            "int16x8x2_t" => quote! { &I16X8X2 },
            "int16x8x3_t" => quote! { &I16X8X3 },
            "int16x8x4_t" => quote! { &I16X8X4 },
            "int32x2_t" => quote! { &I32X2 },
            "int32x2x2_t" => quote! { &I32X2X2 },
            "int32x2x3_t" => quote! { &I32X2X3 },
            "int32x2x4_t" => quote! { &I32X2X4 },
            "int32x4_t" => quote! { &I32X4 },
            "int32x4x2_t" => quote! { &I32X4X2 },
            "int32x4x3_t" => quote! { &I32X4X3 },
            "int32x4x4_t" => quote! { &I32X4X4 },
            "int64x1_t" => quote! { &I64X1 },
            "int64x1x2_t" => quote! { &I64X1X2 },
            "int64x1x3_t" => quote! { &I64X1X3 },
            "int64x1x4_t" => quote! { &I64X1X4 },
            "int64x2_t" => quote! { &I64X2 },
            "int64x2x2_t" => quote! { &I64X2X2 },
            "int64x2x3_t" => quote! { &I64X2X3 },
            "int64x2x4_t" => quote! { &I64X2X4 },
            "uint8x8_t" => quote! { &U8X8 },
            "uint8x4_t" => quote! { &U8X4 },
            "uint8x8x2_t" => quote! { &U8X8X2 },
            "uint8x16x2_t" => quote! { &U8X16X2 },
            "uint8x16x3_t" => quote! { &U8X16X3 },
            "uint8x16x4_t" => quote! { &U8X16X4 },
            "uint8x8x3_t" => quote! { &U8X8X3 },
            "uint8x8x4_t" => quote! { &U8X8X4 },
            "uint8x16_t" => quote! { &U8X16 },
            "uint16x4_t" => quote! { &U16X4 },
            "uint16x4x2_t" => quote! { &U16X4X2 },
            "uint16x4x3_t" => quote! { &U16X4X3 },
            "uint16x4x4_t" => quote! { &U16X4X4 },
            "uint16x8_t" => quote! { &U16X8 },
            "uint16x8x2_t" => quote! { &U16X8X2 },
            "uint16x8x3_t" => quote! { &U16X8X3 },
            "uint16x8x4_t" => quote! { &U16X8X4 },
            "uint32x2_t" => quote! { &U32X2 },
            "uint32x2x2_t" => quote! { &U32X2X2 },
            "uint32x2x3_t" => quote! { &U32X2X3 },
            "uint32x2x4_t" => quote! { &U32X2X4 },
            "uint32x4_t" => quote! { &U32X4 },
            "uint32x4x2_t" => quote! { &U32X4X2 },
            "uint32x4x3_t" => quote! { &U32X4X3 },
            "uint32x4x4_t" => quote! { &U32X4X4 },
            "uint64x1_t" => quote! { &U64X1 },
            "uint64x1x2_t" => quote! { &U64X1X2 },
            "uint64x1x3_t" => quote! { &U64X1X3 },
            "uint64x1x4_t" => quote! { &U64X1X4 },
            "uint64x2_t" => quote! { &U64X2 },
            "uint64x2x2_t" => quote! { &U64X2X2 },
            "uint64x2x3_t" => quote! { &U64X2X3 },
            "uint64x2x4_t" => quote! { &U64X2X4 },
            "float16x2_t" => quote! { &F16X2 },
            "float16x4_t" => quote! { &F16X4 },
            "float16x4x2_t" => quote! { &F16X4X2 },
            "float16x4x3_t" => quote! { &F16X4X3 },
            "float16x4x4_t" => quote! { &F16X4X4 },
            "float16x8_t" => quote! { &F16X8 },
            "float16x8x2_t" => quote! { &F16X8X2 },
            "float16x8x3_t" => quote! { &F16X8X3 },
            "float16x8x4_t" => quote! { &F16X8X4 },
            "float32x2_t" => quote! { &F32X2 },
            "float32x2x2_t" => quote! { &F32X2X2 },
            "float32x2x3_t" => quote! { &F32X2X3 },
            "float32x2x4_t" => quote! { &F32X2X4 },
            "float32x4_t" => quote! { &F32X4 },
            "float32x4x2_t" => quote! { &F32X4X2 },
            "float32x4x3_t" => quote! { &F32X4X3 },
            "float32x4x4_t" => quote! { &F32X4X4 },
            "float64x1_t" => quote! { &F64X1 },
            "float64x1x2_t" => quote! { &F64X1X2 },
            "float64x1x3_t" => quote! { &F64X1X3 },
            "float64x1x4_t" => quote! { &F64X1X4 },
            "float64x2_t" => quote! { &F64X2 },
            "float64x2x2_t" => quote! { &F64X2X2 },
            "float64x2x3_t" => quote! { &F64X2X3 },
            "float64x2x4_t" => quote! { &F64X2X4 },
            "poly8x8_t" => quote! { &POLY8X8 },
            "poly8x8x2_t" => quote! { &POLY8X8X2 },
            "poly8x8x3_t" => quote! { &POLY8X8X3 },
            "poly8x8x4_t" => quote! { &POLY8X8X4 },
            "poly8x16x2_t" => quote! { &POLY8X16X2 },
            "poly8x16x3_t" => quote! { &POLY8X16X3 },
            "poly8x16x4_t" => quote! { &POLY8X16X4 },
            "p64" => quote! { &P64 },
            "poly64x1_t" => quote! { &POLY64X1 },
            "poly64x2_t" => quote! { &POLY64X2 },
            "poly8x16_t" => quote! { &POLY8X16 },
            "poly16x4_t" => quote! { &POLY16X4 },
            "poly16x4x2_t" => quote! { &P16X4X2 },
            "poly16x4x3_t" => quote! { &P16X4X3 },
            "poly16x4x4_t" => quote! { &P16X4X4 },
            "poly16x8_t" => quote! { &POLY16X8 },
            "poly16x8x2_t" => quote! { &P16X8X2 },
            "poly16x8x3_t" => quote! { &P16X8X3 },
            "poly16x8x4_t" => quote! { &P16X8X4 },
            "poly64x1x2_t" => quote! { &P64X1X2 },
            "poly64x1x3_t" => quote! { &P64X1X3 },
            "poly64x1x4_t" => quote! { &P64X1X4 },
            "poly64x2x2_t" => quote! { &P64X2X2 },
            "poly64x2x3_t" => quote! { &P64X2X3 },
            "poly64x2x4_t" => quote! { &P64X2X4 },
            "p128" => quote! { &P128 },

            "v16i8" => quote! { &v16i8 },
            "v8i16" => quote! { &v8i16 },
            "v4i32" => quote! { &v4i32 },
            "v2i64" => quote! { &v2i64 },
            "v16u8" => quote! { &v16u8 },
            "v8u16" => quote! { &v8u16 },
            "v4u32" => quote! { &v4u32 },
            "v2u64" => quote! { &v2u64 },
            "v8f16" => quote! { &v8f16 },
            "v4f32" => quote! { &v4f32 },
            "v2f64" => quote! { &v2f64 },

            // Generic types
            "T" => quote! { &GENERICT },
            "U" => quote! { &GENERICU },

            s => panic!("unsupported type: \"{s}\""),
        },
        syn::Type::Ptr(syn::TypePtr {
            ref elem,
            ref mutability,
            ..
        })
        | syn::Type::Reference(syn::TypeReference {
            ref elem,
            ref mutability,
            ..
        }) => {
            // Both pointers and references can have a mut token (*mut and &mut)
            if mutability.is_some() {
                let tokens = to_type(elem);
                quote! { &Type::MutPtr(#tokens) }
            } else {
                // If they don't (*const or &) then they are "const"
                let tokens = to_type(elem);
                quote! { &Type::ConstPtr(#tokens) }
            }
        }

        syn::Type::Slice(_) => panic!("unsupported slice"),
        syn::Type::Array(_) => panic!("unsupported array"),
        syn::Type::Tuple(_) => quote! { &TUPLE },
        syn::Type::Never(_) => quote! { &NEVER },
        _ => panic!("unsupported type"),
    }
}

fn extract_path_ident(path: &syn::Path) -> syn::Ident {
    if path.leading_colon.is_some() {
        panic!("unsupported leading colon in path")
    }
    if path.segments.len() != 1 {
        panic!("unsupported path that needs name resolution")
    }
    match path.segments.first().expect("segment not found").arguments {
        syn::PathArguments::None => {}
        _ => panic!("unsupported path that has path arguments"),
    }
    path.segments
        .first()
        .expect("segment not found")
        .ident
        .clone()
}

fn walk(root: &Path, files: &mut Vec<(syn::File, String)>) {
    for file in root.read_dir().unwrap() {
        let file = file.unwrap();
        if file.file_type().unwrap().is_dir() {
            walk(&file.path(), files);
            continue;
        }
        let path = file.path();
        if path.extension().and_then(std::ffi::OsStr::to_str) != Some("rs") {
            continue;
        }

        if path.file_name().and_then(std::ffi::OsStr::to_str) == Some("test.rs") {
            continue;
        }

        let mut contents = String::new();
        File::open(&path)
            .unwrap_or_else(|_| panic!("can't open file at path: {}", path.display()))
            .read_to_string(&mut contents)
            .expect("failed to read file to string");

        files.push((
            syn::parse_str::<syn::File>(&contents).expect("failed to parse"),
            path.display().to_string(),
        ));
    }
}

fn find_instrs(attrs: &[syn::Attribute]) -> Vec<String> {
    struct AssertInstr {
        instr: Option<String>,
    }

    // A small custom parser to parse out the instruction in `assert_instr`.
    //
    // TODO: should probably just reuse `Invoc` from the `assert-instr-macro`
    // crate.
    impl syn::parse::Parse for AssertInstr {
        fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
            let _ = input.parse::<syn::Meta>().unwrap();
            let _ = input.parse::<Token![,]>().unwrap();

            match input.parse::<syn::Ident>() {
                Ok(ident) if ident == "assert_instr" => {}
                _ => {
                    while !input.is_empty() {
                        // consume everything
                        drop(input.parse::<proc_macro2::TokenStream>());
                    }
                    return Ok(Self { instr: None });
                }
            }

            let instrs;
            parenthesized!(instrs in input);

            let mut instr = String::new();
            while !instrs.is_empty() {
                if let Ok(lit) = instrs.parse::<syn::LitStr>() {
                    instr.push_str(&lit.value());
                } else if let Ok(ident) = instrs.call(syn::Ident::parse_any) {
                    instr.push_str(&ident.to_string());
                } else if instrs.parse::<Token![.]>().is_ok() {
                    instr.push('.');
                } else if instrs.parse::<Token![,]>().is_ok() {
                    // consume everything remaining
                    drop(instrs.parse::<proc_macro2::TokenStream>());
                    break;
                } else {
                    return Err(input.error("failed to parse instruction"));
                }
            }
            Ok(Self { instr: Some(instr) })
        }
    }

    attrs
        .iter()
        .filter_map(|a| {
            if let syn::Meta::List(ref l) = a.meta {
                if l.path.is_ident("cfg_attr") {
                    Some(l)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .filter_map(|l| syn::parse2::<AssertInstr>(l.tokens.clone()).unwrap().instr)
        .collect()
}

fn find_target_feature(attrs: &[syn::Attribute]) -> Option<syn::Lit> {
    attrs
        .iter()
        .flat_map(|a| {
            #[allow(clippy::collapsible_if)]
            if let syn::Meta::List(ref l) = a.meta {
                if l.path.is_ident("target_feature") {
                    if let Ok(l) =
                        syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated
                            .parse2(l.tokens.clone())
                    {
                        return l;
                    }
                }
            }
            syn::punctuated::Punctuated::new()
        })
        .find_map(|m| match m {
            syn::Meta::NameValue(i) if i.path.is_ident("enable") => {
                if let syn::Expr::Lit(lit) = i.value {
                    Some(lit.lit)
                } else {
                    None
                }
            }
            _ => None,
        })
}

fn find_doc(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .filter_map(|a| {
            #[allow(clippy::collapsible_if)]
            if let syn::Meta::NameValue(ref l) = a.meta {
                if l.path.is_ident("doc") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(ref s),
                        ..
                    }) = l.value
                    {
                        return Some(s.value());
                    }
                }
            }
            None
        })
        .collect()
}

fn find_required_const(name: &str, attrs: &[syn::Attribute]) -> Vec<usize> {
    attrs
        .iter()
        .filter_map(|a| {
            if let syn::Meta::List(ref l) = a.meta {
                Some(l)
            } else {
                None
            }
        })
        .flat_map(|l| {
            if l.path.segments[0].ident == name {
                syn::parse2::<RustcArgsRequiredConst>(l.tokens.clone())
                    .unwrap()
                    .args
            } else {
                Vec::new()
            }
        })
        .collect()
}

struct RustcArgsRequiredConst {
    args: Vec<usize>,
}

impl syn::parse::Parse for RustcArgsRequiredConst {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let list = syn::punctuated::Punctuated::<syn::LitInt, Token![,]>::parse_terminated(input)?;
        Ok(Self {
            args: list
                .into_iter()
                .map(|a| a.base10_parse::<usize>())
                .collect::<syn::Result<_>>()?,
        })
    }
}
