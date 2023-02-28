use quote::{quote, format_ident};
use syn::{FnArg, ReturnType, ItemFn, Signature};
use crate::parser::{DiffItem, Activity, Mode};
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

pub(crate) fn primal_fnc(item: &DiffItem) -> TokenStream {
    let (sig, body) = (&item.primal, &item.block);
    let PrimalSig {
        ident, inputs, output
    } = sig;

    let sig = quote!(fn #ident(#(#inputs,)*) #output);

    quote!(
        #[autodiff_into]
        #sig
        #body
    )
}

fn as_ref_mut(arg: &FnArg, mutability: Option<bool>) -> FnArg {
    arg.clone()
}

pub(crate) fn adjoint_fnc(item: &DiffItem) -> TokenStream {
    let mut res_inputs: Vec<FnArg> = Vec::new();
    let mut outputs = Vec::new();

    let PrimalSig {
        ident, inputs, output
    } = &item.primal;

    for (input, activity) in inputs.iter().zip(item.params.iter()) {
        res_inputs.push(input.clone());

        match (item.header.mode, activity, is_ref_mut(&input)) {
            (Mode::Forward, Activity::Duplicated, Some(true)) => 
                res_inputs.push(as_ref_mut(&input, Some(true))),
            (Mode::Forward, Activity::Duplicated, None) => outputs.push(input.clone()),
            (Mode::Reverse, Activity::Duplicated, Some(false)) =>
                res_inputs.push(as_ref_mut(&input, Some(true))),
            (Mode::Reverse, Activity::Duplicated, Some(true)) =>
                res_inputs.push(as_ref_mut(&input, Some(false))),
            (Mode::Reverse, Activity::Active, None) => outputs.push(input.clone()),
            _ => {}
        }
    }

    let ident = format_ident!("grad_{}", ident);
    let output = quote!(-> #(#outputs,)*);

    let sig = quote!(fn #ident(#(#res_inputs,)*) #output);

    quote!()
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
