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
    assert!(files.len() > 0);

    let mut functions = Vec::new();
    for &mut (ref mut file, ref path) in files.iter_mut() {
        for item in file.items.drain(..) {
            match item {
                syn::Item::Fn(f) => functions.push((f, path)),
                _ => {}
            }
        }
    }
    assert!(functions.len() > 0);

    functions.retain(|&(ref f, _)| {
        match f.vis {
            syn::Visibility::Public(_) => {}
            _ => return false,
        }
        if f.unsafety.is_none() {
            return false;
        }
        true
    });
    assert!(functions.len() > 0);

    let input = proc_macro2::TokenStream::from(input);

    let functions = functions
        .iter()
        .map(|&(ref f, path)| {
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
            let target_feature = match find_target_feature(&f.attrs) {
                Some(i) => my_quote! { Some(#i) },
                None => my_quote! { None },
            };
            my_quote! {
                Function {
                    name: stringify!(#name),
                    arguments: &[#(#arguments),*],
                    ret: #ret,
                    target_feature: #target_feature,
                    instrs: &[#(stringify!(#instrs)),*],
                    file: stringify!(#path),
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
            "__m128" => my_quote! { &M128 },
            "__m128d" => my_quote! { &M128D },
            "__m128i" => my_quote! { &M128I },
            "__m256" => my_quote! { &M256 },
            "__m256d" => my_quote! { &M256D },
            "__m256i" => my_quote! { &M256I },
            "__m64" => my_quote! { &M64 },
            "bool" => my_quote! { &BOOL },
            "f32" => my_quote! { &F32 },
            "f64" => my_quote! { &F64 },
            "i16" => my_quote! { &I16 },
            "i32" => my_quote! { &I32 },
            "i64" => my_quote! { &I64 },
            "i8" => my_quote! { &I8 },
            "u16" => my_quote! { &U16 },
            "u32" => my_quote! { &U32 },
            "u64" => my_quote! { &U64 },
            "u8" => my_quote! { &U8 },
            "CpuidResult" => my_quote! { &CPUID },
            s => panic!("unspported type: {}", s),
        },
        syn::Type::Ptr(syn::TypePtr { ref elem, .. })
        | syn::Type::Reference(syn::TypeReference { ref elem, .. }) => {
            let tokens = to_type(&elem);
            my_quote! { &Type::Ptr(#tokens) }
        }
        syn::Type::Slice(_) => panic!("unsupported slice"),
        syn::Type::Array(_) => panic!("unsupported array"),
        syn::Type::Tuple(_) => my_quote! { &TUPLE },
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
            continue
        }

        let mut contents = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();

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
        .filter_map(|a| a.interpret_meta())
        .filter_map(|a| match a {
            syn::Meta::List(i) => {
                if i.ident == "target_feature" {
                    Some(i.nested)
                } else {
                    None
                }
            }
            _ => None,
        })
        .flat_map(|list| list)
        .filter_map(|nested| {
            match nested {
                syn::NestedMeta::Meta(m) => Some(m),
                syn::NestedMeta::Literal(_) => None,
            }
        })
        .filter_map(|m| {
            match m {
                syn::Meta::NameValue(i) => {
                    if i.ident == "enable" {
                        Some(i.lit)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .next()
}
