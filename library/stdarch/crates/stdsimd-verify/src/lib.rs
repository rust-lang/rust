extern crate proc_macro;
extern crate proc_macro2;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use proc_macro::TokenStream;

#[proc_macro]
pub fn x86_functions(input: TokenStream) -> TokenStream {
    functions(input, &["core_arch/src/x86", "core_arch/src/x86_64"])
}

#[proc_macro]
pub fn arm_functions(input: TokenStream) -> TokenStream {
    functions(input, &["core_arch/src/arm", "core_arch/src/aarch64"])
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
        for item in file.items.drain(..) {
            if let syn::Item::Fn(f) = item {
                functions.push((f, path))
            }
        }
    }
    assert!(!functions.is_empty());

    functions.retain(|&(ref f, _)| {
        if let syn::Visibility::Public(_) = f.vis {
            if f.unsafety.is_some() {
                return true;
            }
        }
        false
    });
    assert!(!functions.is_empty());

    let input = proc_macro2::TokenStream::from(input);

    let functions = functions
        .iter()
        .map(|&(ref f, path)| {
            let name = &f.ident;
            // println!("{}", name);
            let mut arguments = Vec::new();
            for input in f.decl.inputs.iter() {
                let ty = match *input {
                    syn::FnArg::Captured(ref c) => &c.ty,
                    _ => panic!("invalid argument on {}", name),
                };
                arguments.push(to_type(ty));
            }
            let ret = match f.decl.output {
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
            let required_const = find_required_const(&f.attrs);
            quote! {
                Function {
                    name: stringify!(#name),
                    arguments: &[#(#arguments),*],
                    ret: #ret,
                    target_feature: #target_feature,
                    instrs: &[#(stringify!(#instrs)),*],
                    file: stringify!(#path),
                    required_const: &[#(#required_const),*],
                }
            }
        })
        .collect::<Vec<_>>();

    let ret = quote! { #input: &[Function] = &[#(#functions),*]; };
    // println!("{}", ret);
    ret.into()
}

fn to_type(t: &syn::Type) -> proc_macro2::TokenStream {
    match *t {
        syn::Type::Path(ref p) => match extract_path_ident(&p.path).to_string().as_ref() {
            // x86 ...
            "__m128" => quote! { &M128 },
            "__m128d" => quote! { &M128D },
            "__m128i" => quote! { &M128I },
            "__m256" => quote! { &M256 },
            "__m256d" => quote! { &M256D },
            "__m256i" => quote! { &M256I },
            "__m512" => quote! { &M512 },
            "__m512d" => quote! { &M512D },
            "__m512i" => quote! { &M512I },
            "__mmask16" => quote! { &MMASK16 },
            "__m64" => quote! { &M64 },
            "bool" => quote! { &BOOL },
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
            "u8" => quote! { &U8 },
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
            "int16x8_t" => quote! { &I16X8 },
            "int32x2_t" => quote! { &I32X2 },
            "int32x4_t" => quote! { &I32X4 },
            "int64x1_t" => quote! { &I64X1 },
            "int64x2_t" => quote! { &I64X2 },
            "uint8x8_t" => quote! { &U8X8 },
            "uint8x8x2_t" => quote! { &U8X8X2 },
            "uint8x16x2_t" => quote! { &U8X16X2 },
            "uint8x16x3_t" => quote! { &U8X16X3 },
            "uint8x16x4_t" => quote! { &U8X16X4 },
            "uint8x8x3_t" => quote! { &U8X8X3 },
            "uint8x8x4_t" => quote! { &U8X8X4 },
            "uint8x16_t" => quote! { &U8X16 },
            "uint16x4_t" => quote! { &U16X4 },
            "uint16x8_t" => quote! { &U16X8 },
            "uint32x2_t" => quote! { &U32X2 },
            "uint32x4_t" => quote! { &U32X4 },
            "uint64x1_t" => quote! { &U64X1 },
            "uint64x2_t" => quote! { &U64X2 },
            "float32x2_t" => quote! { &F32X2 },
            "float32x4_t" => quote! { &F32X4 },
            "float64x1_t" => quote! { &F64X1 },
            "float64x2_t" => quote! { &F64X2 },
            "poly8x8_t" => quote! { &POLY8X8 },
            "poly8x8x2_t" => quote! { &POLY8X8X2 },
            "poly8x8x3_t" => quote! { &POLY8X8X3 },
            "poly8x8x4_t" => quote! { &POLY8X8X4 },
            "poly8x16x2_t" => quote! { &POLY8X16X2 },
            "poly8x16x3_t" => quote! { &POLY8X16X3 },
            "poly8x16x4_t" => quote! { &POLY8X16X4 },
            "poly64x1_t" => quote! { &POLY64X1 },
            "poly64x2_t" => quote! { &POLY64X2 },
            "poly8x16_t" => quote! { &POLY8X16 },
            "poly16x4_t" => quote! { &POLY16X4 },
            "poly16x8_t" => quote! { &POLY16X8 },

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

            s => panic!("unspported type: \"{}\"", s),
        },
        syn::Type::Ptr(syn::TypePtr { ref elem, .. })
        | syn::Type::Reference(syn::TypeReference { ref elem, .. }) => {
            let tokens = to_type(&elem);
            quote! { &Type::Ptr(#tokens) }
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
    match path
        .segments
        .first()
        .expect("segment not found")
        .value()
        .arguments
    {
        syn::PathArguments::None => {}
        _ => panic!("unsupported path that has path arguments"),
    }
    path.segments
        .first()
        .expect("segment not found")
        .value()
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
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }

        if path.file_name().and_then(|s| s.to_str()) == Some("test.rs") {
            continue;
        }

        let mut contents = String::new();
        File::open(&path)
            .expect(&format!("can't open file at path: {}", path.display()))
            .read_to_string(&mut contents)
            .expect("failed to read file to string");

        files.push((
            syn::parse_str::<syn::File>(&contents).expect("failed to parse"),
            path.display().to_string(),
        ));
    }
}

fn find_instrs(attrs: &[syn::Attribute]) -> Vec<syn::Ident> {
    attrs
        .iter()
        .filter_map(|a| a.interpret_meta())
        .filter_map(|a| match a {
            syn::Meta::List(i) => {
                if i.ident == "cfg_attr" {
                    i.nested.into_iter().nth(1)
                } else {
                    None
                }
            }
            _ => None,
        })
        .filter_map(|nested| match nested {
            syn::NestedMeta::Meta(syn::Meta::List(i)) => {
                if i.ident == "assert_instr" {
                    i.nested.into_iter().next()
                } else {
                    None
                }
            }
            _ => None,
        })
        .filter_map(|nested| match nested {
            syn::NestedMeta::Meta(syn::Meta::Word(i)) => Some(i),
            _ => None,
        })
        .collect()
}

fn find_target_feature(attrs: &[syn::Attribute]) -> Option<syn::Lit> {
    attrs
        .iter()
        .flat_map(|a| {
            if let Some(a) = a.interpret_meta() {
                if let syn::Meta::List(i) = a {
                    if i.ident == "target_feature" {
                        return i.nested;
                    }
                }
            }
            syn::punctuated::Punctuated::new()
        })
        .filter_map(|nested| match nested {
            syn::NestedMeta::Meta(m) => Some(m),
            syn::NestedMeta::Literal(_) => None,
        })
        .filter_map(|m| match m {
            syn::Meta::NameValue(ref i) if i.ident == "enable" => Some(i.clone().lit),
            _ => None,
        })
        .next()
}

fn find_required_const(attrs: &[syn::Attribute]) -> Vec<usize> {
    attrs
        .iter()
        .flat_map(|a| {
            if a.path.segments[0].ident == "rustc_args_required_const" {
                syn::parse::<RustcArgsRequiredConst>(a.tts.clone().into())
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
    #[cfg_attr(feature = "cargo-clippy", allow(clippy::cast_possible_truncation))]
    fn parse(input: syn::parse::ParseStream) -> syn::parse::Result<Self> {
        let content;
        parenthesized!(content in input);
        let list =
            syn::punctuated::Punctuated::<syn::LitInt, Token![,]>::parse_terminated(&content)?;
        Ok(Self {
            args: list.into_iter().map(|a| a.value() as usize).collect(),
        })
    }
}
