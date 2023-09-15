use crate::parser::{is_ref_mut, PrimalSig};
use crate::parser::{Activity, DiffItem, Mode};
use proc_macro2::TokenStream;
use proc_macro_error::abort;
use quote::{format_ident, quote};
use syn::{parse_quote, FnArg, Ident, Pat, ReturnType, Type};

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
    let body = item.block.clone().map(|x| quote!(#x)).unwrap_or_else(|| {
        let header_fnc = &item.header.name;
        //let primal_wrapper = format_ident!("primal_{}", item.primal.ident);
        //item.primal.ident = primal_wrapper.clone();
        let inputs = item.primal.inputs.iter().map(|x| only_ident(x)).collect::<Vec<_>>();

        quote!({
            #header_fnc(#(#inputs,)*)
        })
    });

    let sig = &item.primal;
    let PrimalSig { ident, inputs, output } = sig;

    let ident =
        if item.block.is_some() { ident.clone() } else { format_ident!("primal_{}", ident) };

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
        FnArg::Typed(t) => match &*t.ty {
            Type::Reference(t) => *t.elem.clone(),
            x => x.clone(),
        },
    }
}

fn as_ref_mut(arg: &FnArg, name: &str, mutable: bool) -> FnArg {
    match arg {
        FnArg::Receiver(_) => {
            let name = format_ident!("{}_self", name);
            if mutable { parse_quote!(#name: &mut Self) } else { parse_quote!(#name: &Self) }
        }
        FnArg::Typed(t) => {
            let inner = match &*t.ty {
                Type::Reference(t) => &t.elem,
                _ => panic!(""), // should not be reachable, as we checked mutability before
            };

            let pat_name = match &*t.pat {
                Pat::Ident(x) => &x.ident,
                _ => panic!(""),
            };

            let name = format_ident!("{}_{}", name, pat_name);
            if mutable { parse_quote!(#name: &mut #inner) } else { parse_quote!(#name: &#inner) }
        }
    }
}

pub(crate) fn adjoint_fnc(item: &DiffItem) -> TokenStream {
    let mut res_inputs: Vec<FnArg> = Vec::new();
    let mut add_inputs: Vec<FnArg> = Vec::new();
    let out_type = match &item.primal.output {
        ReturnType::Type(_, x) => Some(*x.clone()),
        _ => None,
    };

    let mut outputs = if item.header.ret_act == Activity::Duplicated {
        vec![out_type.clone().unwrap()]
    } else {
        vec![]
    };

    let PrimalSig { ident, inputs, .. } = &item.primal;

    for (input, activity) in inputs.iter().zip(item.params.iter()) {
        res_inputs.push(input.clone());

        match (item.header.mode, activity, is_ref_mut(&input)) {
            (Mode::Forward, Activity::Duplicated|Activity::DuplicatedNoNeed, Some(true)) => {
                res_inputs.push(as_ref_mut(&input, "grad", true));
                add_inputs.push(as_ref_mut(&input, "grad", true));
            }
            (Mode::Forward, Activity::Duplicated|Activity::DuplicatedNoNeed, Some(false)) => {
                res_inputs.push(as_ref_mut(&input, "dual", false));
                add_inputs.push(as_ref_mut(&input, "dual", false));
                out_type.clone().map(|x| outputs.push(x));
            }
            (Mode::Forward, Activity::Duplicated, None) => outputs.push(only_type(&input)),
            (Mode::Reverse, Activity::Duplicated, Some(false)) => {
                res_inputs.push(as_ref_mut(&input, "grad", true));
                add_inputs.push(as_ref_mut(&input, "grad", true));
            }
            (Mode::Reverse, Activity::Duplicated | Activity::DuplicatedNoNeed, Some(true)) => {
                res_inputs.push(as_ref_mut(&input, "grad", false));
                add_inputs.push(as_ref_mut(&input, "grad", false));
            }
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
        }
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
        }
        _ => quote!(-> (#(#outputs,)*)),
    };

    let sig = quote!(fn #adjoint_ident(#(#res_inputs,)*) #output);
    let inputs = inputs
        .iter()
        .map(|x| match x {
            FnArg::Typed(ty) => {
                let pat = &ty.pat;
                quote!(#pat)
            }
            FnArg::Receiver(_) => quote!(self),
        })
        .collect::<Vec<_>>();
    let add_inputs = add_inputs
        .iter()
        .map(|x| match x {
            FnArg::Typed(ty) => {
                let pat = &ty.pat;
                quote!(#pat)
            }
            FnArg::Receiver(_) => quote!(self),
        })
        .collect::<Vec<_>>();

    let call_ident = match item.block.is_some() {
        false => {
            let ident = format_ident!("primal_{}", ident);
            if item.header.name.segments.first().unwrap().ident == "Self" {
                quote!(Self::#ident)
            } else {
                quote!(#ident)
            }
        }
        true => quote!(#ident),
    };

    let body = quote!({
        std::hint::black_box((#call_ident(#(#inputs,)*), #(#add_inputs,)*));

        std::hint::black_box(unsafe { std::mem::zeroed() })
    });
    let header = generate_header(&item);

    quote!(
        #header
        #sig
        #body
    )
}
