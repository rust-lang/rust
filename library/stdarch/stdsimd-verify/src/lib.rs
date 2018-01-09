#![feature(proc_macro)]

extern crate proc_macro2;
extern crate proc_macro;
#[macro_use]
extern crate quote;
extern crate syn;

use std::path::Path;
use std::fs::File;
use std::io::Read;

use proc_macro::TokenStream;
use quote::Tokens;

macro_rules! my_quote {
    ($($t:tt)*) => (quote_spanned!(proc_macro2::Span::call_site() => $($t)*))
}

#[proc_macro]
pub fn x86_functions(input: TokenStream) -> TokenStream {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = dir.parent().unwrap();
    let root = root.join("coresimd/src/x86");

    let mut files = Vec::new();
    walk(&root, &mut files);

    let mut functions = Vec::new();
    for file in files {
        for item in file.items {
            match item {
                syn::Item::Fn(f) => functions.push(f),
                _ => {}
            }
        }
    }

    functions.retain(|f| {
        match f.vis {
            syn::Visibility::Public(_) => {}
            _ => return false,
        }
        if f.unsafety.is_none() {
            return false;
        }
        f.attrs
            .iter()
            .filter_map(|a| a.interpret_meta())
            .any(|a| match a {
                syn::Meta::NameValue(i) => i.ident == "target_feature",
                _ => false,
            })
    });

    let input = proc_macro2::TokenStream::from(input);

    let functions = functions
        .iter()
        .map(|f| {
            let name = f.ident;
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
                syn::ReturnType::Default => my_quote! { None },
                syn::ReturnType::Type(_, ref t) => {
                    let ty = to_type(t);
                    my_quote! { Some(#ty) }
                }
            };
            let instrs = find_instrs(&f.attrs);
            let target_feature = find_target_feature(f.ident, &f.attrs);
            my_quote! {
                Function {
                    name: stringify!(#name),
                    arguments: &[#(#arguments),*],
                    ret: #ret,
                    target_feature: #target_feature,
                    instrs: &[#(stringify!(#instrs)),*],
                }
            }
        })
        .collect::<Vec<_>>();

    let ret = my_quote! { #input: &[Function] = &[#(#functions),*]; };
    // println!("{}", ret);
    ret.into()
}

fn to_type(t: &syn::Type) -> Tokens {
    match *t {
        syn::Type::Path(ref p) => match extract_path_ident(&p.path).as_ref() {
            "__m128" => my_quote! { &F32x4 },
            "__m128i" => my_quote! { &I8x16 },
            "__m256i" => my_quote! { &I8x32 },
            "__m64" => my_quote! { &I8x8 },
            "bool" => my_quote! { &BOOL },
            "f32" => my_quote! { &F32 },
            "f32x4" => my_quote! { &F32x4 },
            "f32x8" => my_quote! { &F32x8 },
            "f64" => my_quote! { &F64 },
            "f64x2" => my_quote! { &F64x2 },
            "f64x4" => my_quote! { &F64x4 },
            "i16" => my_quote! { &I16 },
            "i16x16" => my_quote! { &I16x16 },
            "i16x4" => my_quote! { &I16x4 },
            "i16x8" => my_quote! { &I16x8 },
            "i32" => my_quote! { &I32 },
            "i32x2" => my_quote! { &I32x2 },
            "i32x4" => my_quote! { &I32x4 },
            "i32x8" => my_quote! { &I32x8 },
            "i64" => my_quote! { &I64 },
            "i64x2" => my_quote! { &I64x2 },
            "i64x4" => my_quote! { &I64x4 },
            "i8" => my_quote! { &I8 },
            "i8x16" => my_quote! { &I8x16 },
            "i8x32" => my_quote! { &I8x32 },
            "i8x8" => my_quote! { &I8x8 },
            "u16x4" => my_quote! { &U16x4 },
            "u16x8" => my_quote! { &U16x8 },
            "u32" => my_quote! { &U32 },
            "u32x2" => my_quote! { &U32x2 },
            "u32x4" => my_quote! { &U32x4 },
            "u32x8" => my_quote! { &U32x8 },
            "u64" => my_quote! { &U64 },
            "u64x2" => my_quote! { &U64x2 },
            "u64x4" => my_quote! { &U64x4 },
            "u8" => my_quote! { &U8 },
            "u16" => my_quote! { &U16 },
            "u8x16" => my_quote! { &U8x16 },
            "u8x32" => my_quote! { &U8x32 },
            "u16x16" => my_quote! { &U16x16 },
            "u8x8" => my_quote! { &U8x8 },
            s => panic!("unspported type: {}", s),
        },
        syn::Type::Ptr(syn::TypePtr { ref elem, .. })
        | syn::Type::Reference(syn::TypeReference { ref elem, .. }) => {
            let tokens = to_type(&elem);
            my_quote! { &Type::Ptr(#tokens) }
        }
        syn::Type::Slice(_) => panic!("unsupported slice"),
        syn::Type::Array(_) => panic!("unsupported array"),
        syn::Type::Tuple(_) => panic!("unsupported tup"),
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
    match path.segments.first().unwrap().value().arguments {
        syn::PathArguments::None => {}
        _ => panic!("unsupported path that has path arguments"),
    }
    path.segments.first().unwrap().value().ident
}

fn walk(root: &Path, files: &mut Vec<syn::File>) {
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

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();

        files.push(
            syn::parse_str::<syn::File>(&contents).expect("failed to parse"),
        );
    }
}

fn find_instrs(attrs: &[syn::Attribute]) -> Vec<syn::Ident> {
    attrs
        .iter()
        .filter_map(|a| a.interpret_meta())
        .filter_map(|a| match a {
            syn::Meta::List(i) => {
                if i.ident == "cfg_attr" {
                    i.nested.into_iter().next()
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

fn find_target_feature(
    name: syn::Ident, attrs: &[syn::Attribute]
) -> syn::Lit {
    attrs
        .iter()
        .filter_map(|a| a.interpret_meta())
        .filter_map(|a| match a {
            syn::Meta::NameValue(i) => {
                if i.ident == "target_feature" {
                    Some(i.lit)
                } else {
                    None
                }
            }
            _ => None,
        })
        .next()
        .expect(&format!("failed to find target_feature for {}", name))
}
