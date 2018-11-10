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
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = dir.parent().unwrap();

    let mut files = Vec::new();
    walk(&root.join("../coresimd/x86"), &mut files);
    walk(&root.join("../coresimd/x86_64"), &mut files);
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
            let target_feature = match find_target_feature(&f.attrs) {
                Some(i) => quote! { Some(#i) },
                None => quote! { None },
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
        syn::Type::Path(ref p) => {
            match extract_path_ident(&p.path).to_string().as_ref() {
                "__m128" => quote! { &M128 },
                "__m128d" => quote! { &M128D },
                "__m128i" => quote! { &M128I },
                "__m256" => quote! { &M256 },
                "__m256d" => quote! { &M256D },
                "__m256i" => quote! { &M256I },
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
                "u8" => quote! { &U8 },
                "CpuidResult" => quote! { &CPUID },
                s => panic!("unspported type: {}", s),
            }
        }
        syn::Type::Ptr(syn::TypePtr { ref elem, .. })
        | syn::Type::Reference(syn::TypeReference { ref elem, .. }) => {
            let tokens = to_type(&elem);
            quote! { &Type::Ptr(#tokens) }
        }
        syn::Type::Slice(_) => panic!("unsupported slice"),
        syn::Type::Array(_) => panic!("unsupported array"),
        syn::Type::Tuple(_) => quote! { &TUPLE },
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
    path.segments.first().unwrap().value().ident.clone()
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
        .filter_map(|nested| match nested {
            syn::NestedMeta::Meta(m) => Some(m),
            syn::NestedMeta::Literal(_) => None,
        })
        .filter_map(|m| match m {
            syn::Meta::NameValue(i) => {
                if i.ident == "enable" {
                    Some(i.lit)
                } else {
                    None
                }
            }
            _ => None,
        })
        .next()
}

fn find_required_const(attrs: &[syn::Attribute]) -> Vec<usize> {
    attrs
        .iter()
        .filter(|a| a.path.segments[0].ident == "rustc_args_required_const")
        .map(|a| a.tts.clone())
        .map(|a| syn::parse::<RustcArgsRequiredConst>(a.into()).unwrap())
        .flat_map(|a| a.args)
        .collect()
}

struct RustcArgsRequiredConst {
    args: Vec<usize>,
}

impl syn::parse::Parse for RustcArgsRequiredConst {
    fn parse(input: syn::parse::ParseStream) -> syn::parse::Result<Self> {
        let content;
        parenthesized!(content in input);
        let list = syn::punctuated::Punctuated::<syn::LitInt, Token![,]>
            ::parse_terminated(&content)?;
        Ok(RustcArgsRequiredConst {
            args: list.into_iter().map(|a| a.value() as usize).collect(),
        })
    }
}
