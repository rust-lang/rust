use quote::{quote, format_ident};
use syn::{FnArg, ReturnType, ItemFn, Signature, Type, parse_quote, Pat, Ident};
use crate::parser::{DiffItem, Activity, Mode};
use proc_macro_error::abort;
use proc_macro2::TokenStream;
use crate::{parser, parser::{PrimalSig, is_ref_mut}};

pub(crate) fn generate_header(item: &DiffItem) -> TokenStream {
    let mode = match item.header.mode {
        Mode::Forward => format_ident!("Forward"),
        Mode::Reverse => format_ident!("Reverse"),
    };
    let ret_act = item.header.ret_act.to_ident();
    let param_act = item.params.iter().map(|x| x.to_ident());

    quote!(#[autodiff_into(#mode, #ret_act, #( #param_act, )*)])
}

pub(crate) fn primal_fnc(item: &mut DiffItem) -> TokenStream {
    // construct body of primal if not given
    let body = item.block.clone().map(|x| quote!(x)).unwrap_or_else(|| {
        let header_fnc = &item.header.name;
        //let primal_wrapper = format_ident!("primal_{}", item.primal.ident);
        //item.primal.ident = primal_wrapper.clone();
        let inputs = item.primal.inputs.iter().map(|x| only_ident(x)).collect::<Vec<_>>();

        quote!({
            #header_fnc(#(#inputs,)*)
        })
    });

    let sig = &item.primal;
    let PrimalSig {
        ident, inputs, output
    } = sig;

    let ident = if item.block.is_some() {
        ident.clone()
    } else {
        format_ident!("primal_{}", ident)
    };

    let sig = quote!(fn #ident(#(#inputs,)*) #output);

    quote!(
        #[autodiff_into]
        #sig
        #body
    )
}

fn only_ident(arg: &FnArg) -> Ident {
    match arg {
        FnArg::Receiver(_) => format_ident!("self"),
        FnArg::Typed(t) => match &*t.pat {
            Pat::Ident(ident) => ident.ident.clone(),
            _ => panic!(""),
        },
    }
}

fn only_type(arg: &FnArg) -> Type {
    match arg {
        FnArg::Receiver(_) => parse_quote!(Self),
        FnArg::Typed(t) => {
            match &*t.ty {
                Type::Reference(t) => *t.elem.clone(),
                x => x.clone(),
            }
        }
    }
}

fn as_ref_mut(arg: &FnArg, name: &str, mutable: bool) -> FnArg {
    match arg {
        FnArg::Receiver(_) => {
            let name = format_ident!("{}_self", name);
            if mutable {
                parse_quote!(#name: &mut Self)
            } else {
                parse_quote!(#name: &Self)
            }
        },
        FnArg::Typed(t) => {
            let inner = match &*t.ty {
                Type::Reference(t) => &t.elem,
                _ => panic!("") // should not be reachable, as we checked mutability before
            };

            let pat_name = match &*t.pat {
                Pat::Ident(x) => &x.ident,
                _ => panic!(""),
            };

            let name = format_ident!("{}_{}", name, pat_name);
            if mutable {
                parse_quote!(#name: &mut #inner)
            } else {
                parse_quote!(#name: &#inner)
            }
        }
    }
}

pub(crate) fn adjoint_fnc(item: &DiffItem) -> TokenStream {
    let mut res_inputs: Vec<FnArg> = Vec::new();
    let mut add_inputs: Vec<FnArg> = Vec::new();
    let mut outputs: Vec<Type> = Vec::new();
    let out_type = 
        match &item.primal.output {
            ReturnType::Type(_, x) => Some(*x.clone()),
            _ => None,
        };

    let PrimalSig {
        ident, inputs, output
    } = &item.primal;

    for (input, activity) in inputs.iter().zip(item.params.iter()) {
        res_inputs.push(input.clone());

        match (item.header.mode, activity, is_ref_mut(&input)) {
            (Mode::Forward, Activity::Duplicated, Some(true)) => {
                res_inputs.push(as_ref_mut(&input, "grad", true));
                add_inputs.push(as_ref_mut(&input, "grad", true));
            },
            (Mode::Forward, Activity::Duplicated, Some(false)) => {
                res_inputs.push(as_ref_mut(&input, "grad", false));
                add_inputs.push(as_ref_mut(&input, "grad", false));
                outputs.push(out_type.clone().unwrap());
            },
            (Mode::Forward, Activity::Duplicated, None) => outputs.push(only_type(&input)),
            (Mode::Reverse, Activity::Duplicated, Some(false)) => {
                res_inputs.push(as_ref_mut(&input, "grad", true));
                add_inputs.push(as_ref_mut(&input, "grad", true));
            },
            (Mode::Reverse, Activity::Duplicated | Activity::DuplicatedNoNeed, Some(true)) => {
                res_inputs.push(as_ref_mut(&input, "grad", false));
                add_inputs.push(as_ref_mut(&input, "grad", false));
            },
            (Mode::Reverse, Activity::Active, None) => outputs.push(only_type(&input)),
            _ => {}
        }
    }

    match (item.header.mode, item.header.ret_act) {
        (Mode::Reverse, Activity::Active) => {
            let t: FnArg = match &item.primal.output {
                ReturnType::Type(_, ty) => parse_quote!(tang_y: #ty),
                _ => panic!(""),
            };
            res_inputs.push(t.clone());
            add_inputs.push(t);
        },
        _ => {}
    }

    // for adjoint function -> take header if primal
    //                      -> take ident of primal function 
    let adjoint_ident = if item.block.is_some() {
        if let Some(ident) = item.header.name.get_ident() {
            ident.clone()
        } else {
            abort!(
                item.header.name,
                "not a function name";
                help = "`#[autodiff]` function name should be a single word instead of path"
            );
        }
    } else {
        item.primal.ident.clone()
    };

    let output = match outputs.len() {
        0 => quote!(),
        1 => {
            let output = outputs.first().unwrap();

            quote!(-> #output)
        },
        _ => quote!(-> (#(#outputs,)*)),
    };

    let sig = quote!(fn #adjoint_ident(#(#res_inputs,)*) #output);
    let inputs = inputs.iter().map(|x| match x {
        FnArg::Typed(ty) => { let pat = &ty.pat; quote!(#pat) },
        FnArg::Receiver(_) => quote!(self),
    }).collect::<Vec<_>>();
    let add_inputs = add_inputs.iter().map(|x| match x {
        FnArg::Typed(ty) => { let pat = &ty.pat; quote!(#pat) },
        FnArg::Receiver(_) => quote!(self),
    }).collect::<Vec<_>>();

    let call_ident = match item.block.is_some() {
        false => {
            let ident = format_ident!("primal_{}", ident);
            if item.header.name.segments.first().unwrap().ident == "Self" {
                quote!(Self::#ident)
            } else {
                quote!(#ident)
            }
        },
        true => quote!(#ident),
    };

    let body = quote!({
        std::hint::black_box((#call_ident(#(#inputs,)*), #(#add_inputs,)*));

        unsafe { std::mem::zeroed() }
    });
    let header = generate_header(&item);

    quote!(
        #header
        #sig
        #body
    )
}

//pub(crate) fn generate_body(token: TokenStream, item: &AutoDiffItem) -> (TokenStream, TokenStream) {
//    let mut fn_args = Vec::new();
//    let mut add_args = Vec::new();
//
//    let mut it_args = item.sig.inputs.iter();
//    for act in &item.params {
//        fn_args.push(it_args.next().unwrap());
//
//        match act {
//            Activity::Duplicated|Activity::DuplicatedNoNeed => add_args.push(it_args.next().unwrap()),
//            _ => {}
//        }
//    }
//
//    if item.header.mode == Mode::Reverse && item.header.ret_act == Activity::Active {
//        let rem_args = it_args.collect::<Vec<_>>();
//
//        if rem_args.len() > 0 {
//            add_args.push(rem_args[0]);
//        }
//    }
//
//    let fn_args_name = fn_args.iter().map(|x| match x {
//        FnArg::Receiver(_) => quote!(self),
//        FnArg::Typed(t) => {
//            let tmp = &t.pat;
//            quote!(#tmp)
//        }
//    });
//    let add_args_name = add_args.iter().map(|x| match x {
//        FnArg::Receiver(_) => quote!(self),
//        FnArg::Typed(t) => {
//            let tmp = &t.pat;
//            quote!(#tmp)
//        }
//    });
//
//    let (fnc_source, fn_name) = if item.block.is_none() {
//        let fn_name = &item.header.name;
//        let fn_name_wrapper = format_ident!("_diff_{}", item.sig.ident);
//        let fn_args_name = fn_args_name.clone();
//        let fn_name_call = match item.sig.inputs.first() {
//            Some(FnArg::Receiver(_)) => quote!(Self::#fn_name_wrapper),
//            _ => quote!(#fn_name_wrapper),
//        };
//
//        // estimate return type on last variable (which is the adjoint)
//        let ret = if item.header.mode == Mode::Reverse && item.header.ret_act == Activity::Active {
//            let last = match fn_args.last().unwrap(){
//                FnArg::Typed(t) => &t.ty,
//                _ => panic!(""),
//            };
//
//            quote!(-> #last)
//        } else {
//            quote!()
//        };
//
//        (
//            quote!(
//                fn #fn_name_wrapper(#( #fn_args, )*) #ret {
//                    #fn_name(#( #fn_args_name, )*)
//                }
//            ),
//            fn_name_call
//        )
//    } else {
//        let mut iitem = syn::parse2::<ItemFn>(token).unwrap();
//        let (params, _) = parser::strip_sig_attributes(iitem.sig.inputs.iter().collect(), false, &item.header);
//        iitem.sig.inputs = params.into_iter().collect();
//
//        let fn_name = &iitem.sig.ident;
//        (quote!(#iitem), quote!(#fn_name))
//    };
//
//    let ret = match item.sig.output {
//        ReturnType::Type(_, _) => quote!(unsafe { std::mem::zeroed() }),
//        _ => quote!(),
//    };
//    let tmp = fn_args_name.clone();
//
//    (quote!(
//        std::hint::black_box((#fn_name(#( #fn_args_name, )*), #( &#add_args_name, )* #( &#tmp, )*));
//
//        #ret
//    ), quote!(#fnc_source))
//}
