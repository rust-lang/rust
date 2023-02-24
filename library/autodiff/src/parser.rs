use syn::{Item, ForeignItemFn, Block, parse::Parser, punctuated::Punctuated, Path, Token, Signature, Ident, FnArg, Attribute, Type, ReturnType, parse_quote, Generics};
use proc_macro2::TokenStream;
use proc_macro_error::abort;
use quote::format_ident;

#[derive(Debug)]
pub struct PrimalSig {
    ident: Ident,
    generics: Generics,
    inputs: Vec<FnArg>,
    output: ReturnType,
}

#[derive(Debug)]
pub struct DiffItem {
    pub(crate) header: Header,
    //pub(crate) params: Vec<Activity>,
    //pub(crate) primal: PrimalSig,
    //pub(crate) block: Option<Box<Block>>,
}

impl Default for DiffItem {
    fn default() -> Self {
        DiffItem {
            header: Header {
                name: parse_quote!(""),
                mode: Mode::Forward,
                ret_act: Activity::Const,
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Mode {
    Forward,
    Reverse
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Activity {
    Const,
    Active,
    Duplicated,
    DuplicatedNoNeed,
}

impl Activity {
    fn from_header(name: Option<&Ident>) -> Activity {
        if name.is_none() {
            return Activity::Const;
        }

        match name.unwrap().to_string().as_str() {
            "Const" => Activity::Const,
            "Active" => Activity::Active,
            "Duplicated" => Activity::Duplicated,
            "DuplicatedNoNeed" => Activity::DuplicatedNoNeed,
            _ => {
                abort!(
                    name,
                    "unknown activity";
                    help = "`#[autodiff]` should use activities (Const|Active|Duplicated|DuplicatedNoNeed)"
                );
            }
        }
    }

    fn from_inline(name: Attribute) -> Activity {
        let name = name.path.segments.first().unwrap();
        match name.ident.to_string().as_str() {
            "const" => Activity::Const,
            "active" => Activity::Active,
            "dup" => Activity::Duplicated,
            "dup_noneed" => Activity::DuplicatedNoNeed,
            _ => {
                abort!(
                    name,
                    "unknown activity";
                    help = "`#[autodiff]` should use activities (const|active|dup|dup_noneed)"
                );
            }
        }
    }

    pub(crate) fn to_ident(&self) -> Ident {
        format_ident!("{}", match self {
            Activity::Const => "Const",
            Activity::Active => "Active",
            Activity::Duplicated => "Duplicated",
            Activity::DuplicatedNoNeed => "DuplicatedNoNeed",
        })
    }
}

#[derive(Debug)]
pub(crate) struct Header {
    pub name: Path,
    pub mode: Mode,
    pub ret_act: Activity,
}

impl Header {
    fn from_params(name: &Path, mode: Option<&Ident>, ret_activity: Option<&Ident>) -> Self {
        // parse mode and return activity
        let mode = mode.map(|x| match x.to_string().as_str() {
            "forward" | "Forward" => Mode::Forward,
            "reverse" | "Reverse" => Mode::Reverse,
            _ => {
                abort!(
                    mode,
                    "should be forward or reverse";
                    help = "`#[autodiff]` modes should be either forward or reverse"
                );
            }
        }).unwrap_or(Mode::Forward);
        let ret_act = Activity::from_header(ret_activity);

        // check for invalid mode and return activity combinations
        match (mode, ret_act) {
            (Mode::Forward, Activity::Active) =>
                abort!(
                    ret_activity,
                    "active return for forward mode";
                    help = "`#[autodiff]` return should be Const, Duplicated or DuplicatedNoNeed in forward mode"
                ),
            (Mode::Reverse, Activity::Duplicated | Activity::DuplicatedNoNeed) =>
                abort!(
                    ret_activity,
                    "duplicated return for reverse mode";
                    help = "`#[autodiff]` return should be Const or Active in reverse mode"
                ),

            _ => {},
        }

        Header {
            name: name.clone(),
            mode,
            ret_act,
        }
    }

    fn parse(args: TokenStream) -> (Header, Vec<Activity>) {
        let args_parsed: Vec<_> = match Punctuated::<Path, Token![,]>::parse_terminated
            .parse(args.clone().into()) {
                Ok(x) => x.into_iter().collect(),
                Err(err) => 
                    abort!(
                        args,
                        "duplicated return for reverse mode";
                        help = "`#[autodiff]` return should be Const or Active in reverse mode"
                    )
            };

        match &args_parsed[..] {
            [name] => (Self::from_params(&name, None, None), vec![]),
            [name, mode] => (Self::from_params(&name, Some(&mode.get_ident().unwrap()), None), vec![]),
            [name, mode, ret_act, rem @ ..] => {
                let params = Self::from_params(&name, Some(&mode.get_ident().unwrap()), Some(&ret_act.get_ident().unwrap()));
                let rem = rem.into_iter()
                    .map(|x| x.get_ident().unwrap())
                    .map(|x| Activity::from_header(Some(x)))
                    .map(|x| match (params.mode, x) {
                        (Mode::Forward, Activity::Active) => {
                            abort!(
                                args,
                                "active argument in forward mode";
                                help = "`#[autodiff]` forward mode should be either Const, Duplicated"
                            );
                        },
                        (_, x) => x,
                    })
                    .collect();

                (params, rem)
            },
            _ => {
                abort!(
                    args,
                    "please specify the autodiff function";
                    help = "`#[autodiff]` needs a function name for primal or adjoint"
                );
            }
        }
    }
}

pub(crate) fn strip_sig_attributes(args: Vec<&FnArg>, do_skip: bool, _header: &Header) -> (Vec<FnArg>, Vec<Activity>) {
    let mut args = args.into_iter().cloned().collect::<Vec<_>>();
    let mut arg_it = args.iter_mut();
    let mut acts = Vec::new();
    let mut skip = false;

    while let Some(arg) = arg_it.next() {
        if skip && do_skip {
            skip = false;
            continue;
        }

        let mut attrs = match arg {
            FnArg::Typed(pat) => pat.attrs.drain(..),
            FnArg::Receiver(pat) => pat.attrs.drain(..),
        };
        let act = attrs.next().map(Activity::from_inline).unwrap_or(Activity::Const);
        match act {
            Activity::Duplicated | Activity::DuplicatedNoNeed => { skip = true; }
            _ => {}
        }
        acts.push(act);
    }

    // remove last activity because it belongs to return type
    //if header.mode == Mode::Reverse && header.ret_act == Activity::Active {
    //    dbg!(&acts);
    //    acts.pop();
    //}

    (args, acts)
}

//
//fn is_ref_mut(t: &Type) -> bool {
//    match t {
//        Type::Reference(t) => t.mutability.is_some(),
//        _ => {
//            abort!(
//                t,
//                "not a reference";
//                help = "`#[autodiff]` arguments should be references"
//            );
//        }
//    }
//}
//
//fn check_output(arg: &FnArg) -> bool {
//    match arg {
//        FnArg::Receiver(x) => x.mutability.is_some(),
//        FnArg::Typed(t) => is_ref_mut(&t.ty),
//    }
//}
//
//fn dup_arg_with_name_mut(arg: &FnArg, name: &str, mutable: bool) -> FnArg {
//    match arg {
//        FnArg::Receiver(_) => {
//            let name = format_ident!("{}_self", name);
//            if mutable {
//                parse_quote!(#name: &mut Self)
//            } else {
//                parse_quote!(#name: &Self)
//            }
//        },
//        FnArg::Typed(t) => {
//
//            let inner = match &*t.ty {
//                Type::Reference(t) => &t.elem,
//                _ => panic!("") // should not be reachable, as we checked mutability before
//            };
//
//            let pat_name = match &*t.pat {
//                Pat::Ident(x) => &x.ident,
//                _ => panic!(""),
//            };
//
//            let name = format_ident!("{}_{}", name, pat_name);
//            if mutable {
//                parse_quote!(#name: &mut #inner)
//            } else {
//                parse_quote!(#name: &#inner)
//            }
//        }
//    }
//}
//
//fn ret_arg(arg: &FnArg) -> Type {
//    match arg {
//        FnArg::Receiver(_) => parse_quote!(Self),
//        FnArg::Typed(t) => {
//            match &*t.ty {
//                Type::Reference(t) => *t.elem.clone(),
//                _ => panic!(""),
//            }
//        }
//    }
//}
//
//fn create_target_signature_forward(mut sig: Signature, act: &Vec<Activity>, ret_act: &Activity) -> Signature {
//    let mut inputs = Vec::new();
//    let mut outputs = Vec::new();
//    for (p, a) in sig.inputs.iter().zip(act.into_iter()) {
//        let is_output = check_output(p);
//
//        if !is_output {
//            inputs.push(p.clone());
//
//            if *a != Activity::Const {
//                inputs.push(dup_arg_with_name_mut(&p, "adj", false));
//            }
//            
//            if *ret_act != Activity::Const {
//                match sig.output {
//                    ReturnType::Type(_, ref ty) => outputs.push(ty.clone()),
//                    _ => panic!(""),
//                }
//            }
//        } else {
//            inputs.push(p.clone());
//
//            if *a != Activity::Const {
//                inputs.push(dup_arg_with_name_mut(&p, "d", true));
//            }
//        }
//    }
//
//    sig.inputs = inputs.into_iter().collect();
//
//    if *ret_act != Activity::Const {
//        let ret_ty = match sig.output {
//            ReturnType::Type(_, t) => t,
//            _ => {
//                abort!(
//                    sig.output,
//                    "no return type";
//                    help = "`#[autodiff]` specified duplicated activity but function has not return"
//                );
//            }
//        };
//
//        sig.output = if *ret_act == Activity::Duplicated {
//            parse_quote!(-> (#ret_ty, #( #outputs, )*))
//        } else {
//            if outputs.len() > 1 {
//                parse_quote!(-> (#( #outputs, )*))
//            } else {
//                parse_quote!(-> #( #outputs )*)
//            }
//        };
//    }
//        
//    sig
//}
//
//fn create_target_signature_reverse(mut sig: Signature, act: &Vec<Activity>, ret_act: &Activity) -> Signature {
//    let mut inputs = Vec::new();
//    let mut outputs = Vec::new();
//    for (p, a) in sig.inputs.iter().zip(act.into_iter()) {
//        let is_output = check_output(p);
//
//        if !is_output {
//            inputs.push(p.clone());
//
//            match a {
//                Activity::Active => {
//                    outputs.push(ret_arg(&p));
//                },
//                Activity::Duplicated | Activity::DuplicatedNoNeed => inputs.push(dup_arg_with_name_mut(&p, "d", true)),
//                _ => {}
//            }
//        } else {
//            inputs.push(p.clone());
//
//            if *a != Activity::Const {
//                inputs.push(dup_arg_with_name_mut(&p, "adj", false));
//            }
//        }
//    }
//
//    match sig.output {
//        ReturnType::Type(_, typ) => {
//            inputs.push(parse_quote!(ret_adj: #typ));
//        },
//        _ => {}
//    }
//
//    sig.inputs = inputs.into_iter().collect();
//
//    sig.output = if *ret_act == Activity::Active {
//        match outputs.len() {
//            0 => parse_quote!(),
//            1 => parse_quote!(-> #( #outputs )*),
//            _ => parse_quote!(-> (#( #outputs, )*))
//        }
//    } else {
//        parse_quote!()
//    };
//        
//    sig
//}
pub(crate) fn parse(args: TokenStream, input: TokenStream) -> DiffItem {
    // first parse function
    let (_attrs, _, mut sig, block) = match syn::parse2::<Item>(input) {
        Ok(Item::Fn(item)) => (item.attrs, item.vis, item.sig, Some(item.block)),
        Ok(Item::Verbatim(x)) => {
            match syn::parse2::<ForeignItemFn>(x) {
                Ok(item) => (item.attrs, item.vis, item.sig, None),
                Err(err) => panic!("Could not parse item {}", err)
            }
        },
        Ok(item) => {
            abort!(
                item,
                "item is not a function";
                help = "`#[autodiff]` can only be used on primal or adjoint functions"
            )
        },
        Err(err) => panic!("Could not parse item: {}", err)
    };


    // then parse attributes
    let (header, param_attrs) = Header::parse(args);

    //// strip the function parameters from attribute macros
    let (params, param_attrs2) = strip_sig_attributes(sig.inputs.iter().collect(), !block.is_some(), &header);

    DiffItem::default()

    //sig.inputs = params.into_iter().collect();

    //let mut params = match (param_attrs, param_attrs2) {
    //    (a, _) if !a.is_empty() => a,
    //    (_, b) if !b.is_empty() => b,
    //    _ => Vec::new(),
    //};

    //let sig = match block.is_some() {
    //    true => match header.mode {
    //        Mode::Forward => create_target_signature_forward(sig, &params, &header.ret_act),
    //        Mode::Reverse => create_target_signature_reverse(sig, &params, &header.ret_act)
    //    },
    //    false => sig,
    //};

    //let mut rem_params = sig.inputs.len() - params.iter().map(|x| match x {
    //    Activity::Const | Activity::Active => 1,
    //    Activity::Duplicated | Activity::DuplicatedNoNeed => 2,
    //}).sum::<usize>();

    //// don't add const for return adjoint variable
    //if header.ret_act == Activity::Active && header.mode == Mode::Reverse && rem_params > 0 {
    //    rem_params -= 1;
    //}

    //params.extend((0..rem_params).map(|_| Activity::Const));

    //Item {
    //    header,
    //    params,
    //    sig,
    //    block
    //}
}
