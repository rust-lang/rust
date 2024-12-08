use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::*;
use syn::*;

// We parse the input and emit the output in a single step.
// This field stores the final macro output
struct Newtype(TokenStream);

impl Parse for Newtype {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;

        let body;
        braced!(body in input);

        // Any additional `#[derive]` macro paths to apply
        let mut derive_paths: Vec<Path> = Vec::new();
        let mut debug_format: Option<Lit> = None;
        let mut max = None;
        let mut consts = Vec::new();
        let mut encodable = false;
        let mut ord = false;
        let mut gate_rustc_only = quote! {};
        let mut gate_rustc_only_cfg = quote! { all() };

        attrs.retain(|attr| match attr.path().get_ident() {
            Some(ident) => match &*ident.to_string() {
                "gate_rustc_only" => {
                    gate_rustc_only = quote! { #[cfg(feature = "nightly")] };
                    gate_rustc_only_cfg = quote! { feature = "nightly" };
                    false
                }
                "encodable" => {
                    encodable = true;
                    false
                }
                "orderable" => {
                    ord = true;
                    false
                }
                "max" => {
                    let Meta::NameValue(MetaNameValue { value: Expr::Lit(lit), .. }) = &attr.meta
                    else {
                        panic!("#[max = NUMBER] attribute requires max value");
                    };

                    if let Some(old) = max.replace(lit.lit.clone()) {
                        panic!("Specified multiple max: {old:?}");
                    }

                    false
                }
                "debug_format" => {
                    let Meta::NameValue(MetaNameValue { value: Expr::Lit(lit), .. }) = &attr.meta
                    else {
                        panic!("#[debug_format = FMT] attribute requires a format");
                    };

                    if let Some(old) = debug_format.replace(lit.lit.clone()) {
                        panic!("Specified multiple debug format options: {old:?}");
                    }

                    false
                }
                _ => true,
            },
            _ => true,
        });

        loop {
            // We've parsed everything that the user provided, so we're done
            if body.is_empty() {
                break;
            }

            // Otherwise, we are parsing a user-defined constant
            let const_attrs = body.call(Attribute::parse_outer)?;
            body.parse::<Token![const]>()?;
            let const_name: Ident = body.parse()?;
            body.parse::<Token![=]>()?;
            let const_val: Expr = body.parse()?;
            body.parse::<Token![;]>()?;
            consts.push(quote! { #(#const_attrs)* #vis const #const_name: #name = #name::from_u32(#const_val); });
        }

        let debug_format =
            debug_format.unwrap_or_else(|| Lit::Str(LitStr::new("{}", Span::call_site())));

        // shave off 256 indices at the end to allow space for packing these indices into enums
        let max = max.unwrap_or_else(|| Lit::Int(LitInt::new("0xFFFF_FF00", Span::call_site())));

        let encodable_impls = if encodable {
            quote! {
                #gate_rustc_only
                impl<D: ::rustc_serialize::Decoder> ::rustc_serialize::Decodable<D> for #name {
                    fn decode(d: &mut D) -> Self {
                        Self::from_u32(d.read_u32())
                    }
                }
                #gate_rustc_only
                impl<E: ::rustc_serialize::Encoder> ::rustc_serialize::Encodable<E> for #name {
                    fn encode(&self, e: &mut E) {
                        e.emit_u32(self.as_u32());
                    }
                }
            }
        } else {
            quote! {}
        };

        if ord {
            derive_paths.push(parse_quote!(Ord));
            derive_paths.push(parse_quote!(PartialOrd));
        }

        let step = if ord {
            quote! {
                #gate_rustc_only
                impl ::std::iter::Step for #name {
                    #[inline]
                    fn steps_between(start: &Self, end: &Self) -> (usize, Option<usize>) {
                        <usize as ::std::iter::Step>::steps_between(
                            &Self::index(*start),
                            &Self::index(*end),
                        )
                    }

                    #[inline]
                    fn forward_checked(start: Self, u: usize) -> Option<Self> {
                        Self::index(start).checked_add(u).map(Self::from_usize)
                    }

                    #[inline]
                    fn backward_checked(start: Self, u: usize) -> Option<Self> {
                        Self::index(start).checked_sub(u).map(Self::from_usize)
                    }
                }
            }
        } else {
            quote! {}
        };

        let debug_impl = quote! {
            impl ::std::fmt::Debug for #name {
                fn fmt(&self, fmt: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    write!(fmt, #debug_format, self.as_u32())
                }
            }
        };

        Ok(Self(quote! {
            #(#attrs)*
            #[derive(Clone, Copy, PartialEq, Eq, Hash, #(#derive_paths),*)]
            #[cfg_attr(#gate_rustc_only_cfg, rustc_layout_scalar_valid_range_end(#max))]
            #[cfg_attr(#gate_rustc_only_cfg, rustc_pass_by_value)]
            #vis struct #name {
                private_use_as_methods_instead: u32,
            }

            #(#consts)*

            impl #name {
                /// Maximum value the index can take, as a `u32`.
                #vis const MAX_AS_U32: u32  = #max;

                /// Maximum value the index can take.
                #vis const MAX: Self = Self::from_u32(#max);

                /// Zero value of the index.
                #vis const ZERO: Self = Self::from_u32(0);

                /// Creates a new index from a given `usize`.
                ///
                /// # Panics
                ///
                /// Will panic if `value` exceeds `MAX`.
                #[inline]
                #vis const fn from_usize(value: usize) -> Self {
                    assert!(value <= (#max as usize));
                    // SAFETY: We just checked that `value <= max`.
                    unsafe {
                        Self::from_u32_unchecked(value as u32)
                    }
                }

                /// Creates a new index from a given `u32`.
                ///
                /// # Panics
                ///
                /// Will panic if `value` exceeds `MAX`.
                #[inline]
                #vis const fn from_u32(value: u32) -> Self {
                    assert!(value <= #max);
                    // SAFETY: We just checked that `value <= max`.
                    unsafe {
                        Self::from_u32_unchecked(value)
                    }
                }

                /// Creates a new index from a given `u16`.
                ///
                /// # Panics
                ///
                /// Will panic if `value` exceeds `MAX`.
                #[inline]
                #vis const fn from_u16(value: u16) -> Self {
                    let value = value as u32;
                    assert!(value <= #max);
                    // SAFETY: We just checked that `value <= max`.
                    unsafe {
                        Self::from_u32_unchecked(value)
                    }
                }

                /// Creates a new index from a given `u32`.
                ///
                /// # Safety
                ///
                /// The provided value must be less than or equal to the maximum value for the newtype.
                /// Providing a value outside this range is undefined due to layout restrictions.
                ///
                /// Prefer using `from_u32`.
                #[inline]
                #vis const unsafe fn from_u32_unchecked(value: u32) -> Self {
                    Self { private_use_as_methods_instead: value }
                }

                /// Extracts the value of this index as a `usize`.
                #[inline]
                #vis const fn index(self) -> usize {
                    self.as_usize()
                }

                /// Extracts the value of this index as a `u32`.
                #[inline]
                #vis const fn as_u32(self) -> u32 {
                    self.private_use_as_methods_instead
                }

                /// Extracts the value of this index as a `usize`.
                #[inline]
                #vis const fn as_usize(self) -> usize {
                    self.as_u32() as usize
                }
            }

            impl std::ops::Add<usize> for #name {
                type Output = Self;

                #[inline]
                fn add(self, other: usize) -> Self {
                    Self::from_usize(self.index() + other)
                }
            }

            impl rustc_index::Idx for #name {
                #[inline]
                fn new(value: usize) -> Self {
                    Self::from_usize(value)
                }

                #[inline]
                fn index(self) -> usize {
                    self.as_usize()
                }
            }

            #step

            impl From<#name> for u32 {
                #[inline]
                fn from(v: #name) -> u32 {
                    v.as_u32()
                }
            }

            impl From<#name> for usize {
                #[inline]
                fn from(v: #name) -> usize {
                    v.as_usize()
                }
            }

            impl From<usize> for #name {
                #[inline]
                fn from(value: usize) -> Self {
                    Self::from_usize(value)
                }
            }

            impl From<u32> for #name {
                #[inline]
                fn from(value: u32) -> Self {
                    Self::from_u32(value)
                }
            }

            #encodable_impls
            #debug_impl
        }))
    }
}

pub fn newtype(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as Newtype);
    input.0.into()
}
