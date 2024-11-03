mod enums;
mod parse;

use std::sync::LazyLock;

use parse::{Invocation, StructuredInput};
use proc_macro as pm;
use proc_macro2::{self as pm2, Span};
use quote::{ToTokens, quote};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{Ident, ItemEnum};

const ALL_FUNCTIONS: &[(Ty, Signature, Option<Signature>, &[&str])] = &[
    (
        // `fn(f32) -> f32`
        Ty::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32] },
        None,
        &[
            "acosf", "acoshf", "asinf", "asinhf", "atanf", "atanhf", "cbrtf", "ceilf", "cosf",
            "coshf", "erff", "exp10f", "exp2f", "expf", "expm1f", "fabsf", "floorf", "j0f", "j1f",
            "lgammaf", "log10f", "log1pf", "log2f", "logf", "rintf", "roundf", "sinf", "sinhf",
            "sqrtf", "tanf", "tanhf", "tgammaf", "truncf",
        ],
    ),
    (
        // `(f64) -> f64`
        Ty::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64] },
        None,
        &[
            "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "ceil", "cos", "cosh",
            "erf", "exp10", "exp2", "exp", "expm1", "fabs", "floor", "j0", "j1", "lgamma", "log10",
            "log1p", "log2", "log", "rint", "round", "sin", "sinh", "sqrt", "tan", "tanh",
            "tgamma", "trunc",
        ],
    ),
    (
        // `(f32, f32) -> f32`
        Ty::F32,
        Signature { args: &[Ty::F32, Ty::F32], returns: &[Ty::F32] },
        None,
        &[
            "atan2f",
            "copysignf",
            "fdimf",
            "fmaxf",
            "fminf",
            "fmodf",
            "hypotf",
            "nextafterf",
            "powf",
            "remainderf",
        ],
    ),
    (
        // `(f64, f64) -> f64`
        Ty::F64,
        Signature { args: &[Ty::F64, Ty::F64], returns: &[Ty::F64] },
        None,
        &[
            "atan2",
            "copysign",
            "fdim",
            "fmax",
            "fmin",
            "fmod",
            "hypot",
            "nextafter",
            "pow",
            "remainder",
        ],
    ),
    (
        // `(f32, f32, f32) -> f32`
        Ty::F32,
        Signature { args: &[Ty::F32, Ty::F32, Ty::F32], returns: &[Ty::F32] },
        None,
        &["fmaf"],
    ),
    (
        // `(f64, f64, f64) -> f64`
        Ty::F64,
        Signature { args: &[Ty::F64, Ty::F64, Ty::F64], returns: &[Ty::F64] },
        None,
        &["fma"],
    ),
    (
        // `(f32) -> i32`
        Ty::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::I32] },
        None,
        &["ilogbf"],
    ),
    (
        // `(f64) -> i32`
        Ty::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::I32] },
        None,
        &["ilogb"],
    ),
    (
        // `(i32, f32) -> f32`
        Ty::F32,
        Signature { args: &[Ty::I32, Ty::F32], returns: &[Ty::F32] },
        None,
        &["jnf"],
    ),
    (
        // `(i32, f64) -> f64`
        Ty::F64,
        Signature { args: &[Ty::I32, Ty::F64], returns: &[Ty::F64] },
        None,
        &["jn"],
    ),
    (
        // `(f32, i32) -> f32`
        Ty::F32,
        Signature { args: &[Ty::F32, Ty::I32], returns: &[Ty::F32] },
        None,
        &["scalbnf", "ldexpf"],
    ),
    (
        // `(f64, i64) -> f64`
        Ty::F64,
        Signature { args: &[Ty::F64, Ty::I32], returns: &[Ty::F64] },
        None,
        &["scalbn", "ldexp"],
    ),
    (
        // `(f32, &mut f32) -> f32` as `(f32) -> (f32, f32)`
        Ty::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::F32] },
        Some(Signature { args: &[Ty::F32, Ty::MutF32], returns: &[Ty::F32] }),
        &["modff"],
    ),
    (
        // `(f64, &mut f64) -> f64` as  `(f64) -> (f64, f64)`
        Ty::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::F64] },
        Some(Signature { args: &[Ty::F64, Ty::MutF64], returns: &[Ty::F64] }),
        &["modf"],
    ),
    (
        // `(f32, &mut c_int) -> f32` as `(f32) -> (f32, i32)`
        Ty::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::I32] },
        Some(Signature { args: &[Ty::F32, Ty::MutCInt], returns: &[Ty::F32] }),
        &["frexpf", "lgammaf_r"],
    ),
    (
        // `(f64, &mut c_int) -> f64` as `(f64) -> (f64, i32)`
        Ty::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::I32] },
        Some(Signature { args: &[Ty::F64, Ty::MutCInt], returns: &[Ty::F64] }),
        &["frexp", "lgamma_r"],
    ),
    (
        // `(f32, f32, &mut c_int) -> f32` as `(f32, f32) -> (f32, i32)`
        Ty::F32,
        Signature { args: &[Ty::F32, Ty::F32], returns: &[Ty::F32, Ty::I32] },
        Some(Signature { args: &[Ty::F32, Ty::F32, Ty::MutCInt], returns: &[Ty::F32] }),
        &["remquof"],
    ),
    (
        // `(f64, f64, &mut c_int) -> f64` as `(f64, f64) -> (f64, i32)`
        Ty::F64,
        Signature { args: &[Ty::F64, Ty::F64], returns: &[Ty::F64, Ty::I32] },
        Some(Signature { args: &[Ty::F64, Ty::F64, Ty::MutCInt], returns: &[Ty::F64] }),
        &["remquo"],
    ),
    (
        // `(f32, &mut f32, &mut f32)` as `(f32) -> (f32, f32)`
        Ty::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::F32] },
        Some(Signature { args: &[Ty::F32, Ty::MutF32, Ty::MutF32], returns: &[] }),
        &["sincosf"],
    ),
    (
        // `(f64, &mut f64, &mut f64)` as `(f64) -> (f64, f64)`
        Ty::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::F64] },
        Some(Signature { args: &[Ty::F64, Ty::MutF64, Ty::MutF64], returns: &[] }),
        &["sincos"],
    ),
];

const KNOWN_TYPES: &[&str] = &["FTy", "CFn", "CArgs", "CRet", "RustFn", "RustArgs", "RustRet"];

/// A type used in a function signature.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum Ty {
    F16,
    F32,
    F64,
    F128,
    I32,
    CInt,
    MutF16,
    MutF32,
    MutF64,
    MutF128,
    MutI32,
    MutCInt,
}

impl ToTokens for Ty {
    fn to_tokens(&self, tokens: &mut pm2::TokenStream) {
        let ts = match self {
            Ty::F16 => quote! { f16 },
            Ty::F32 => quote! { f32 },
            Ty::F64 => quote! { f64 },
            Ty::F128 => quote! { f128 },
            Ty::I32 => quote! { i32 },
            Ty::CInt => quote! { ::core::ffi::c_int },
            Ty::MutF16 => quote! { &'a mut f16 },
            Ty::MutF32 => quote! { &'a mut f32 },
            Ty::MutF64 => quote! { &'a mut f64 },
            Ty::MutF128 => quote! { &'a mut f128 },
            Ty::MutI32 => quote! { &'a mut i32 },
            Ty::MutCInt => quote! { &'a mut core::ffi::c_int },
        };

        tokens.extend(ts);
    }
}

/// Representation of e.g. `(f32, f32) -> f32`
#[derive(Debug, Clone)]
struct Signature {
    args: &'static [Ty],
    returns: &'static [Ty],
}

/// Combined information about a function implementation.
#[derive(Debug, Clone)]
struct FunctionInfo {
    name: &'static str,
    base_fty: Ty,
    /// Function signature for C implementations
    c_sig: Signature,
    /// Function signature for Rust implementations
    rust_sig: Signature,
}

/// A flat representation of `ALL_FUNCTIONS`.
static ALL_FUNCTIONS_FLAT: LazyLock<Vec<FunctionInfo>> = LazyLock::new(|| {
    let mut ret = Vec::new();

    for (base_fty, rust_sig, c_sig, names) in ALL_FUNCTIONS {
        for name in *names {
            let api = FunctionInfo {
                name,
                base_fty: *base_fty,
                rust_sig: rust_sig.clone(),
                c_sig: c_sig.clone().unwrap_or_else(|| rust_sig.clone()),
            };
            ret.push(api);
        }
    }

    ret.sort_by_key(|item| item.name);
    ret
});

/// Populate an enum with a variant representing function. Names are in upper camel case.
///
/// Applied to an empty enum. Expects one attribute `#[function_enum(BaseName)]` that provides
/// the name of the `BaseName` enum.
#[proc_macro_attribute]
pub fn function_enum(attributes: pm::TokenStream, tokens: pm::TokenStream) -> pm::TokenStream {
    let item = syn::parse_macro_input!(tokens as ItemEnum);
    let res = enums::function_enum(item, attributes.into());

    match res {
        Ok(ts) => ts,
        Err(e) => e.into_compile_error(),
    }
    .into()
}

/// Create an enum representing all possible base names, with names in upper camel case.
///
/// Applied to an empty enum.
#[proc_macro_attribute]
pub fn base_name_enum(attributes: pm::TokenStream, tokens: pm::TokenStream) -> pm::TokenStream {
    let item = syn::parse_macro_input!(tokens as ItemEnum);
    let res = enums::base_name_enum(item, attributes.into());

    match res {
        Ok(ts) => ts,
        Err(e) => e.into_compile_error(),
    }
    .into()
}

/// Do something for each function present in this crate.
///
/// Takes a callback macro and invokes it multiple times, once for each function that
/// this crate exports. This makes it easy to create generic tests, benchmarks, or other checks
/// and apply it to each symbol.
///
/// Additionally, the `extra` and `fn_extra` patterns can make use of magic identifiers:
///
/// - `MACRO_FN_NAME`: gets replaced with the name of the function on that invocation.
/// - `MACRO_FN_NAME_NORMALIZED`: similar to the above, but removes sufixes so e.g. `sinf` becomes
///   `sin`, `cosf128` becomes `cos`, etc.
///
/// Invoke as:
///
/// ```
/// // Macro that is invoked once per function
/// macro_rules! callback_macro {
///     (
///         // Name of that function
///         fn_name: $fn_name:ident,
///         // The basic float type for this function (e.g. `f32`, `f64`)
///         FTy: $FTy:ty,
///         // Function signature of the C version (e.g. `fn(f32, &mut f32) -> f32`)
///         CFn: $CFn:ty,
///         // A tuple representing the C version's arguments (e.g. `(f32, &mut f32)`)
///         CArgs: $CArgs:ty,
///         // The C version's return type (e.g. `f32`)
///         CRet: $CRet:ty,
///         // Function signature of the Rust version (e.g. `fn(f32) -> (f32, f32)`)
///         RustFn: $RustFn:ty,
///         // A tuple representing the Rust version's arguments (e.g. `(f32,)`)
///         RustArgs: $RustArgs:ty,
///         // The Rust version's return type (e.g. `(f32, f32)`)
///         RustRet: $RustRet:ty,
///         // Attributes for the current function, if any
///         attrs: [$($meta:meta)*]
///         // Extra tokens passed directly (if any)
///         extra: [$extra:ident],
///         // Extra function-tokens passed directly (if any)
///         fn_extra: $fn_extra:expr,
///     ) => { };
/// }
///
/// // All fields except for `callback` are optional.
/// libm_macros::for_each_function! {
///     // The macro to invoke as a callback
///     callback: callback_macro,
///     // Which types to include either as a list (`[CFn, RustFn, RustArgs]`) or "all"
///     emit_types: all,
///     // Functions to skip, i.e. `callback` shouldn't be called at all for these.
///     skip: [sin, cos],
///     // Attributes passed as `attrs` for specific functions. For example, here the invocation
///     // with `sinf` and that with `cosf` will both get `meta1` and `meta2`, but no others will.
///     attributes: [
///         #[meta1]
///         #[meta2]
///         [sinf, cosf],
///     ],
///     // Any tokens that should be passed directly to all invocations of the callback. This can
///     // be used to pass local variables or other things the macro needs access to.
///     extra: [foo],
///     // Similar to `extra`, but allow providing a pattern for only specific functions. Uses
///     // a simplified match-like syntax.
///     fn_extra: match MACRO_FN_NAME {
///         hypot | hypotf => |x| x.hypot(),
///         _ => |x| x,
///     },
/// }
/// ```
#[proc_macro]
pub fn for_each_function(tokens: pm::TokenStream) -> pm::TokenStream {
    let input = syn::parse_macro_input!(tokens as Invocation);

    let res = StructuredInput::from_fields(input)
        .and_then(|mut s_in| validate(&mut s_in).map(|fn_list| (s_in, fn_list)))
        .and_then(|(s_in, fn_list)| expand(s_in, &fn_list));

    match res {
        Ok(ts) => ts.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

/// Check for any input that is structurally correct but has other problems.
///
/// Returns the list of function names that we should expand for.
fn validate(input: &mut StructuredInput) -> syn::Result<Vec<&'static FunctionInfo>> {
    // Collect lists of all functions that are provied as macro inputs in various fields (only,
    // skip, attributes).
    let attr_mentions = input
        .attributes
        .iter()
        .flat_map(|map_list| map_list.iter())
        .flat_map(|attr_map| attr_map.names.iter());
    let only_mentions = input.only.iter().flat_map(|only_list| only_list.iter());
    let fn_extra_mentions =
        input.fn_extra.iter().flat_map(|v| v.keys()).filter(|name| *name != "_");
    let all_mentioned_fns =
        input.skip.iter().chain(only_mentions).chain(attr_mentions).chain(fn_extra_mentions);

    // Make sure that every function mentioned is a real function
    for mentioned in all_mentioned_fns {
        if !ALL_FUNCTIONS_FLAT.iter().any(|func| mentioned == func.name) {
            let e = syn::Error::new(
                mentioned.span(),
                format!("unrecognized function name `{mentioned}`"),
            );
            return Err(e);
        }
    }

    if !input.skip.is_empty() && input.only.is_some() {
        let e = syn::Error::new(
            input.only_span.unwrap(),
            "only one of `skip` or `only` may be specified",
        );
        return Err(e);
    }

    // Construct a list of what we intend to expand
    let mut fn_list = Vec::new();
    for func in ALL_FUNCTIONS_FLAT.iter() {
        let fn_name = func.name;
        // If we have an `only` list and it does _not_ contain this function name, skip it
        if input.only.as_ref().is_some_and(|only| !only.iter().any(|o| o == fn_name)) {
            continue;
        }

        // If there is a `skip` list that contains this function name, skip it
        if input.skip.iter().any(|s| s == fn_name) {
            continue;
        }

        // Run everything else
        fn_list.push(func);
    }

    // Types that the user would like us to provide in the macro
    let mut add_all_types = false;
    for ty in &input.emit_types {
        let ty_name = ty.to_string();
        if ty_name == "all" {
            add_all_types = true;
            continue;
        }

        // Check that all requested types are valid
        if !KNOWN_TYPES.contains(&ty_name.as_str()) {
            let e = syn::Error::new(
                ty_name.span(),
                format!("unrecognized type identifier `{ty_name}`"),
            );
            return Err(e);
        }
    }

    if add_all_types {
        // Ensure that if `all` was specified that nothing else was
        if input.emit_types.len() > 1 {
            let e = syn::Error::new(
                input.emit_types_span.unwrap(),
                "if `all` is specified, no other type identifiers may be given",
            );
            return Err(e);
        }

        // ...and then add all types
        input.emit_types.clear();
        for ty in KNOWN_TYPES {
            let ident = Ident::new(ty, Span::call_site());
            input.emit_types.push(ident);
        }
    }

    if let Some(map) = &input.fn_extra {
        if !map.keys().any(|key| key == "_") {
            // No default provided; make sure every expected function is covered
            let mut fns_not_covered = Vec::new();
            for func in &fn_list {
                if !map.keys().any(|key| key == func.name) {
                    // `name` was not mentioned in the `match` statement
                    fns_not_covered.push(func);
                }
            }

            if !fns_not_covered.is_empty() {
                let e = syn::Error::new(
                    input.fn_extra_span.unwrap(),
                    format!(
                        "`fn_extra`: no default `_` pattern specified and the following \
                         patterns are not covered: {fns_not_covered:#?}"
                    ),
                );
                return Err(e);
            }
        }
    };

    Ok(fn_list)
}

/// Expand our structured macro input into invocations of the callback macro.
fn expand(input: StructuredInput, fn_list: &[&FunctionInfo]) -> syn::Result<pm2::TokenStream> {
    let mut out = pm2::TokenStream::new();
    let default_ident = Ident::new("_", Span::call_site());
    let callback = input.callback;

    for func in fn_list {
        let fn_name = Ident::new(func.name, Span::call_site());

        // Prepare attributes in an `attrs: ...` field
        let meta_field = match &input.attributes {
            Some(attrs) => {
                let meta = attrs
                    .iter()
                    .filter(|map| map.names.contains(&fn_name))
                    .flat_map(|map| &map.meta);
                quote! { attrs: [ #( #meta )* ]  }
            }
            None => pm2::TokenStream::new(),
        };

        // Prepare extra in an `extra: ...` field, running the replacer
        let extra_field = match input.extra.clone() {
            Some(mut extra) => {
                let mut v = MacroReplace::new(func.name);
                v.visit_expr_mut(&mut extra);
                v.finish()?;

                quote! { extra: #extra, }
            }
            None => pm2::TokenStream::new(),
        };

        // Prepare function-specific extra in a `fn_extra: ...` field, running the replacer
        let fn_extra_field = match input.fn_extra {
            Some(ref map) => {
                let mut fn_extra =
                    map.get(&fn_name).or_else(|| map.get(&default_ident)).unwrap().clone();

                let mut v = MacroReplace::new(func.name);
                v.visit_expr_mut(&mut fn_extra);
                v.finish()?;

                quote! { fn_extra: #fn_extra, }
            }
            None => pm2::TokenStream::new(),
        };

        let base_fty = func.base_fty;
        let c_args = &func.c_sig.args;
        let c_ret = &func.c_sig.returns;
        let rust_args = &func.rust_sig.args;
        let rust_ret = &func.rust_sig.returns;

        let mut ty_fields = Vec::new();
        for ty in &input.emit_types {
            let field = match ty.to_string().as_str() {
                "FTy" => quote! { FTy: #base_fty, },
                "CFn" => quote! { CFn: fn( #(#c_args),* ,) -> ( #(#c_ret),* ), },
                "CArgs" => quote! { CArgs: ( #(#c_args),* ,), },
                "CRet" => quote! { CRet: ( #(#c_ret),* ), },
                "RustFn" => quote! { RustFn: fn( #(#rust_args),* ,) -> ( #(#rust_ret),* ), },
                "RustArgs" => quote! { RustArgs: ( #(#rust_args),* ,), },
                "RustRet" => quote! { RustRet: ( #(#rust_ret),* ), },
                _ => unreachable!("checked in validation"),
            };
            ty_fields.push(field);
        }

        let new = quote! {
            #callback! {
                fn_name: #fn_name,
                #( #ty_fields )*
                #meta_field
                #extra_field
                #fn_extra_field
            }
        };

        out.extend(new);
    }

    Ok(out)
}

/// Visitor to replace "magic" identifiers that we allow: `MACRO_FN_NAME` and
/// `MACRO_FN_NAME_NORMALIZED`.
struct MacroReplace {
    fn_name: &'static str,
    /// Remove the trailing `f` or `f128` to make
    norm_name: String,
    error: Option<syn::Error>,
}

impl MacroReplace {
    fn new(name: &'static str) -> Self {
        let norm_name = base_name(name);
        Self { fn_name: name, norm_name: norm_name.to_owned(), error: None }
    }

    fn finish(self) -> syn::Result<()> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn visit_ident_inner(&mut self, i: &mut Ident) {
        let s = i.to_string();
        if !s.starts_with("MACRO") || self.error.is_some() {
            return;
        }

        match s.as_str() {
            "MACRO_FN_NAME" => *i = Ident::new(self.fn_name, i.span()),
            "MACRO_FN_NAME_NORMALIZED" => *i = Ident::new(&self.norm_name, i.span()),
            _ => {
                self.error =
                    Some(syn::Error::new(i.span(), format!("unrecognized meta expression `{s}`")));
            }
        }
    }
}

impl VisitMut for MacroReplace {
    fn visit_ident_mut(&mut self, i: &mut Ident) {
        self.visit_ident_inner(i);
        syn::visit_mut::visit_ident_mut(self, i);
    }
}

/// Return the unsuffixed version of a function name; e.g. `abs` and `absf` both return `abs`,
/// `lgamma_r` and `lgammaf_r` both return `lgamma_r`.
fn base_name(name: &str) -> &str {
    let known_mappings = &[
        ("erff", "erf"),
        ("erf", "erf"),
        ("lgammaf_r", "lgamma_r"),
        ("modff", "modf"),
        ("modf", "modf"),
    ];

    match known_mappings.iter().find(|known| known.0 == name) {
        Some(found) => found.1,
        None => name
            .strip_suffix("f")
            .or_else(|| name.strip_suffix("f16"))
            .or_else(|| name.strip_suffix("f128"))
            .unwrap_or(name),
    }
}
