mod enums;
mod parse;
mod shared;

use parse::{Invocation, StructuredInput};
use proc_macro as pm;
use proc_macro2::{self as pm2, Span};
use quote::{ToTokens, quote};
pub(crate) use shared::{ALL_OPERATIONS, FloatTy, MathOpInfo, Ty};
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{Ident, ItemEnum};

const KNOWN_TYPES: &[&str] = &[
    "FTy", "CFn", "CArgs", "CRet", "RustFn", "RustArgs", "RustRet", "public",
];

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
///         // True if this is part of `libm`'s public API
///         public: $public:expr,
///         // Attributes for the current function, if any
///         attrs: [$($attr:meta),*],
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
///     //
///     // Note that `f16_enabled` and `f128_enabled` will always get emitted regardless of whether
///     // or not this is specified.
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
///         // `ALL_*` magic matchers also work to extract specific types
///         ALL_F64 => |x| x,
///         // The default pattern gets applied to everything that did not match
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
fn validate(input: &mut StructuredInput) -> syn::Result<Vec<&'static MathOpInfo>> {
    // Replace magic mappers with a list of relevant functions.
    if let Some(map) = &mut input.fn_extra {
        for (name, ty) in [
            ("ALL_F16", FloatTy::F16),
            ("ALL_F32", FloatTy::F32),
            ("ALL_F64", FloatTy::F64),
            ("ALL_F128", FloatTy::F128),
        ] {
            let Some(k) = map.keys().find(|key| *key == name) else {
                continue;
            };

            let key = k.clone();
            let val = map.remove(&key).unwrap();

            for op in ALL_OPERATIONS.iter().filter(|op| op.float_ty == ty) {
                map.insert(Ident::new(op.name, key.span()), val.clone());
            }
        }
    }

    // Collect lists of all functions that are provied as macro inputs in various fields (only,
    // skip, attributes).
    let attr_mentions = input
        .attributes
        .iter()
        .flat_map(|map_list| map_list.iter())
        .flat_map(|attr_map| attr_map.names.iter());
    let only_mentions = input.only.iter().flat_map(|only_list| only_list.iter());
    let fn_extra_mentions = input
        .fn_extra
        .iter()
        .flat_map(|v| v.keys())
        .filter(|name| *name != "_");
    let all_mentioned_fns = input
        .skip
        .iter()
        .chain(only_mentions)
        .chain(attr_mentions)
        .chain(fn_extra_mentions);

    // Make sure that every function mentioned is a real function
    for mentioned in all_mentioned_fns {
        if !ALL_OPERATIONS.iter().any(|func| mentioned == func.name) {
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
    for func in ALL_OPERATIONS.iter() {
        let fn_name = func.name;
        // If we have an `only` list and it does _not_ contain this function name, skip it
        if input
            .only
            .as_ref()
            .is_some_and(|only| !only.iter().any(|o| o == fn_name))
        {
            continue;
        }

        // If there is a `skip` list that contains this function name, skip it
        if input.skip.iter().any(|s| s == fn_name) {
            continue;
        }

        // Omit f16 and f128 functions if requested
        if input.skip_f16_f128 && (func.float_ty == FloatTy::F16 || func.float_ty == FloatTy::F128)
        {
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

    if let Some(map) = &input.fn_extra
        && !map.keys().any(|key| key == "_")
    {
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
    };

    Ok(fn_list)
}

/// Expand our structured macro input into invocations of the callback macro.
fn expand(input: StructuredInput, fn_list: &[&MathOpInfo]) -> syn::Result<pm2::TokenStream> {
    let mut out = pm2::TokenStream::new();
    let default_ident = Ident::new("_", Span::call_site());
    let callback = input.callback;

    for func in fn_list {
        let fn_name = Ident::new(func.name, Span::call_site());

        // Prepare attributes in an `attrs: ...` field
        let mut meta_fields = Vec::new();
        if let Some(attrs) = &input.attributes {
            let meta_iter = attrs
                .iter()
                .filter(|map| map.names.contains(&fn_name))
                .flat_map(|map| &map.meta)
                .map(|v| v.into_token_stream());

            meta_fields.extend(meta_iter);
        }

        // Always emit f16 and f128 meta so this doesn't need to be repeated everywhere
        if func.rust_sig.args.contains(&Ty::F16) || func.rust_sig.returns.contains(&Ty::F16) {
            let ts = quote! { cfg(f16_enabled) };
            meta_fields.push(ts);
        }
        if func.rust_sig.args.contains(&Ty::F128) || func.rust_sig.returns.contains(&Ty::F128) {
            let ts = quote! { cfg(f128_enabled) };
            meta_fields.push(ts);
        }

        let meta_field = quote! { attrs: [ #( #meta_fields ),* ], };

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
                let mut fn_extra = map
                    .get(&fn_name)
                    .or_else(|| map.get(&default_ident))
                    .unwrap()
                    .clone();

                let mut v = MacroReplace::new(func.name);
                v.visit_expr_mut(&mut fn_extra);
                v.finish()?;

                quote! { fn_extra: #fn_extra, }
            }
            None => pm2::TokenStream::new(),
        };

        let base_fty = func.float_ty;
        let c_args = &func.c_sig.args;
        let c_ret = &func.c_sig.returns;
        let rust_args = &func.rust_sig.args;
        let rust_ret = &func.rust_sig.returns;
        let public = func.public;

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
                "public" => quote! { public: #public, },
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
        Self {
            fn_name: name,
            norm_name: norm_name.to_owned(),
            error: None,
        }
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
                self.error = Some(syn::Error::new(
                    i.span(),
                    format!("unrecognized meta expression `{s}`"),
                ));
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
impl ToTokens for FloatTy {
    fn to_tokens(&self, tokens: &mut pm2::TokenStream) {
        let ts = match self {
            FloatTy::F16 => quote! { f16 },
            FloatTy::F32 => quote! { f32 },
            FloatTy::F64 => quote! { f64 },
            FloatTy::F128 => quote! { f128 },
        };

        tokens.extend(ts);
    }
}
