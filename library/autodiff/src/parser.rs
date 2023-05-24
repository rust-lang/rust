use proc_macro2::TokenStream;
use proc_macro_error::abort;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_quote, punctuated::Punctuated, Attribute, Block, FnArg, ForeignItemFn,
    Ident, Item, Path, ReturnType, Signature, Token, Type,
};

#[derive(Debug)]
pub struct PrimalSig {
    pub(crate) ident: Ident,
    pub(crate) inputs: Vec<FnArg>,
    pub(crate) output: ReturnType,
}

#[derive(Debug)]
pub struct DiffItem {
    pub(crate) header: Header,
    pub(crate) params: Vec<Activity>,
    pub(crate) primal: PrimalSig,
    pub(crate) block: Option<Box<Block>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum Mode {
    Forward,
    Reverse,
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
        format_ident!(
            "{}",
            match self {
                Activity::Const => "Const",
                Activity::Active => "Active",
                Activity::Duplicated => "Duplicated",
                Activity::DuplicatedNoNeed => "DuplicatedNoNeed",
            }
        )
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
        let mode = mode
            .map(|x| match x.to_string().as_str() {
                "forward" | "Forward" => Mode::Forward,
                "reverse" | "Reverse" => Mode::Reverse,
                _ => {
                    abort!(
                        mode,
                        "should be forward or reverse";
                        help = "`#[autodiff]` modes should be either forward or reverse"
                    );
                }
            })
            .unwrap_or(Mode::Forward);
        let ret_act = Activity::from_header(ret_activity);

        // check for invalid mode and return activity combinations
        match (mode, ret_act) {
            (Mode::Forward, Activity::Active) => abort!(
                ret_activity,
                "active return for forward mode";
                help = "`#[autodiff]` return should be Const, Duplicated or DuplicatedNoNeed in forward mode"
            ),
            (Mode::Reverse, Activity::Duplicated | Activity::DuplicatedNoNeed) => abort!(
                ret_activity,
                "duplicated return for reverse mode";
                help = "`#[autodiff]` return should be Const or Active in reverse mode"
            ),

            _ => {}
        }

        Header { name: name.clone(), mode, ret_act }
    }

    fn parse(args: TokenStream) -> (Header, Vec<Activity>) {
        let args_parsed: Vec<_> =
            match Punctuated::<Path, Token![,]>::parse_terminated.parse(args.clone().into()) {
                Ok(x) => x.into_iter().collect(),
                Err(_) => abort!(
                    args,
                    "duplicated return for reverse mode";
                    help = "`#[autodiff]` return should be Const or Active in reverse mode"
                ),
            };

        match &args_parsed[..] {
            [name] => (Self::from_params(&name, None, None), vec![]),
            [name, mode] => {
                (Self::from_params(&name, Some(&mode.get_ident().unwrap()), None), vec![])
            }
            [name, mode, ret_act, rem @ ..] => {
                let params = Self::from_params(
                    &name,
                    Some(&mode.get_ident().unwrap()),
                    Some(&ret_act.get_ident().unwrap()),
                );
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
            }
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

pub(crate) fn is_ref_mut(t: &FnArg) -> Option<bool> {
    match t {
        FnArg::Receiver(pat) => Some(pat.mutability.is_some()),
        FnArg::Typed(pat) => match &*pat.ty {
            Type::Reference(t) => Some(t.mutability.is_some()),
            _ => None,
        },
    }
}

fn is_scalar(t: &Type) -> bool {
    let t_f32: Type = parse_quote!(f32);
    let t_f64: Type = parse_quote!(f64);
    t == &t_f32 || t == &t_f64
}

fn ret_arg(arg: &FnArg) -> Type {
    match arg {
        FnArg::Receiver(_) => parse_quote!(Self),
        FnArg::Typed(t) => match &*t.ty {
            Type::Reference(t) => *t.elem.clone(),
            x => x.clone(),
        },
    }
}

pub(crate) fn reduce_params(
    mut sig: Signature,
    header_acts: Vec<Activity>,
    is_adjoint: bool,
    header: &Header,
) -> (PrimalSig, Vec<Activity>) {
    let mut args = Vec::new();
    let mut ret = Vec::new();
    let mut acts = Vec::new();
    let mut last_arg: Option<FnArg> = None;

    let mut arg_it = sig.inputs.iter_mut();
    let mut header_acts_it = header_acts.iter();

    while let Some(arg) = arg_it.next() {
        // Compare current with last argument when parsing duplicated rules. This only
        // happens when we parse the signature of adjoint/augmented primal function
        if let Some(prev_arg) = last_arg.take() {
            match (header.mode, is_ref_mut(&prev_arg), is_ref_mut(&arg)) {
                (Mode::Forward, Some(false), Some(true) | None) => abort!(
                    arg,
                    "should be an immutable reference";
                    help = "`#[autodiff]` input parameter should duplicate tangent into second parameter for forward mode"
                ),
                (Mode::Forward, Some(true), Some(false) | None) => abort!(
                    arg,
                    "should be a mutable reference";
                    help = "`#[autodiff]` output parameter should duplicate derivative into second parameter for forward mode"
                ),
                (Mode::Reverse, Some(false), Some(false) | None) => abort!(
                    arg,
                    "should be a mutable reference";
                    help = "`#[autodiff]` input parameter should duplicate derivative into second parameter for reverse mode"
                ),
                (Mode::Reverse, Some(true), Some(true) | None) => abort!(
                    arg,
                    "should be an immutable reference";
                    help = "`#[autodiff]` input parameter should duplicate derivative into second parameter for reverse mode"
                ),
                _ => {}
            }

            continue;
        }

        // parse current attribute macro
        let attrs: Vec<_> = match arg {
            FnArg::Typed(pat) => pat.attrs.drain(..).collect(),
            FnArg::Receiver(pat) => pat.attrs.drain(..).collect(),
        };
        let attr = attrs.first();
        let act: Activity = match (header_acts.is_empty(), attr) {
            (false, None) => header_acts_it.next().map(|x| *x).unwrap_or(Activity::Const),
            (true, Some(x)) => Activity::from_inline(x.clone()),
            (true, None) => Activity::Const,
            _ => {
                abort!(
                    arg,
                    "inline activity";
                    help = "`#[autodiff]` should have activities either specified in header or as inline attributes"
                );
            }
        };

        // compare indirection with activity
        match (header.mode, is_ref_mut(&arg), act) {
            (Mode::Forward, None, Activity::Duplicated) => abort!(
                arg,
                "type not behind reference";
                help = "`#[autodiff]` duplicated types should be behind a reference"
            ),
            (Mode::Forward, Some(false), Activity::DuplicatedNoNeed) => abort!(
                arg,
                "should be mutable reference";
                help = "`#[autodiff]` parameter should be output for DuplicatedNoNeed activity"
            ),
            (Mode::Reverse, Some(_), Activity::Active) => abort!(
                arg,
                "type behind reference";
                help = "`#[autodiff]` active parameter should be concrete in reverse mode"
            ),
            (Mode::Reverse, None, Activity::Duplicated | Activity::DuplicatedNoNeed) => abort!(
                arg,
                "type not behind reference";
                help = "`#[autodiff]` duplicated parameters should be behind reference in reverse mode"
            ),
            (Mode::Reverse, Some(false), Activity::DuplicatedNoNeed) => abort!(
                arg,
                "use duplicated instead";
                help = "`#[autodiff]` input parameter cannot be declared as duplicatednoneed"
            ),
            (Mode::Forward, Some(false), Activity::Duplicated)
                if header.ret_act != Activity::Const =>
            {
                ret.push(ret_arg(&arg))
            }
            (Mode::Reverse, None, Activity::Active) => ret.push(ret_arg(&arg)),
            (Mode::Forward, Some(_), Activity::Duplicated | Activity::DuplicatedNoNeed)
            | (Mode::Reverse, _, Activity::Duplicated | Activity::DuplicatedNoNeed)
                if is_adjoint =>
            {
                last_arg = Some(arg.clone())
            }
            _ => {}
        }

        args.push(arg.clone());
        acts.push(act);
    }

    // if we have adjoint signature and are in forward mode
    // if duplicated -> return type * (n + 1) times
    // if duplicated_no_need -> return type * n times
    // if const -> no return

    // if we have adjoint signature and are in reverse mode
    // if active -> input type * n times
    // construct return type based on mode
    let ret = if is_adjoint {
        let ret_typs = match &sig.output {
            ReturnType::Type(_, ref x) => match &**x {
                Type::Tuple(x) => x.elems.iter().cloned().collect(),
                x => vec![x.clone()],
            },
            ReturnType::Default => vec![],
        };

        match (header.mode, header.ret_act) {
            (Mode::Forward, Activity::Duplicated) => {
                let expected = ret_typs[0].clone();
                let list = vec![expected.clone(); ret.len() + 1];

                if list != ret_typs {
                    let ret = quote!((#(#list,)*));
                    abort!(
                        sig.output,
                        "invalid output";
                        help = format!("`#[autodiff]` expected {}", ret)
                    );
                }

                parse_quote!(-> #expected)
            }
            (Mode::Forward, Activity::DuplicatedNoNeed) => {
                let expected = ret_typs[0].clone();
                let list = vec![expected.clone(); ret.len()];

                if list != ret_typs {
                    let ret = quote!((#(#list,)*));
                    abort!(
                        sig.output,
                        "invalid output";
                        help = format!("`#[autodiff]` expected {}", ret)
                    );
                }

                parse_quote!(-> #expected)
            }
            (Mode::Reverse, Activity::Active) => {
                // tangent of output is latest in parameter list
                let ret_typ = match (args.pop(), acts.pop()) {
                    (Some(x), Some(y)) => {
                        let x = ret_arg(&x);
                        if !is_scalar(&x) {
                            abort!(
                                x,
                                "output tangent not a floating point";
                                help = "`#[autodiff]` the output tangent should be a floating point"
                            );
                        } else if y != Activity::Const {
                            abort!(
                                x,
                                "output tangent not const";
                                help = "`#[autodiff]` the last parameter of an adjoint with active return should be a constant tangent"
                            );
                        } else {
                            parse_quote!(-> #x)
                        }
                    }
                    (None, None) => abort!(
                        sig,
                        "missing output tangent parameter";
                        help = "`#[autodiff]` the last parameter of an adjoint with active return should exist"
                    ),
                    _ => unreachable!(),
                };

                // check that the return tuple confirms with return types
                if ret_typs != ret {
                    let ret = quote!((#(#ret,)*));
                    abort!(
                        sig.output,
                        "invalid output";
                        help = format!("`#[autodiff]` expected {}", ret)
                    )
                }

                ret_typ
            }
            (_, Activity::Const) if ret.len() > 0 => {
                abort!(
                    ret[0],
                    "constant return but more than one return";
                    help = "`#[autodiff]` adjoint should have a return type when active"
                )
            }
            _ => ReturnType::Default,
        }
    } else {
        if header.ret_act != Activity::Const && sig.output == ReturnType::Default {
            abort!(
                sig,
                "no return type";
                help = "`#[autodiff]` non-const return activity but no return type"
            )
        }

        sig.output.clone()
    };

    let sig = if is_adjoint {
        // header is used for calling if we are adjoint
        format_ident!("{}", sig.ident)
    } else {
        sig.ident.clone()
    };

    (PrimalSig { ident: sig, inputs: args, output: ret }, acts)
}

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
    let (_attrs, _, sig, block) = match syn::parse2::<Item>(input) {
        Ok(Item::Fn(item)) => (item.attrs, item.vis, item.sig, Some(item.block)),
        Ok(Item::Verbatim(x)) => match syn::parse2::<ForeignItemFn>(x) {
            Ok(item) => (item.attrs, item.vis, item.sig, None),
            Err(err) => panic!("Could not parse item {}", err),
        },
        Ok(item) => {
            abort!(
                item,
                "item is not a function";
                help = "`#[autodiff]` can only be used on primal or adjoint functions"
            )
        }
        Err(err) => panic!("Could not parse item: {}", err),
    };

    // then parse attributes
    let (header, param_attrs) = Header::parse(args);

    // reduce parameters to primal parameter set
    let (primal, params) = reduce_params(sig, param_attrs, !block.is_some(), &header);

    DiffItem { header, primal, params, block }
}
