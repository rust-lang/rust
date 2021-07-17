use proc_macro2::TokenStream;
use quote::quote;

pub fn early_declared_functions(_input: TokenStream) -> TokenStream {
    let ty_ctx_impl = quote! {
        impl<'tcx> TyCtxt<'tcx> {
            fn bar(self, arg1: Ty) -> RetTy {
                (self.fn_provider.bar)((arg1,))
            }
        }
    };
    let provider_struct = quote! {
        struct FnProvider {
            bar: for<'tcx> fn(_: TyCtxt<'tcx>, _: (Ty1,)) -> RetTy,
        }
    };
    let provider_impl = quote! {
        impl FnProvider {
            fn empty() -> Self {
                FnProvider {
                    bar: |_| bug!("the function `bar` has not been initialized"),
                }
            }
        }
    };
    vec![ty_ctx_impl, provider_struct, provider_impl].into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::early_declared_functions;
    use proc_macro2::TokenStream;
    use quote::quote;

    fn assert_macro(
        macro_fn: impl Fn(TokenStream) -> TokenStream,
        input: TokenStream,
        expect: TokenStream,
    ) {
        assert_eq!(macro_fn(input).to_string(), expect.to_string())
    }
    #[test]
    fn empty() {
        assert_macro(early_declared_functions, quote! {}, quote! {});
    }

    #[test]
    fn single_fn_and_arg() {
        assert_macro(
            early_declared_functions,
            quote! {fn bar(arg1: Ty) -> RetTy;},
            quote! {
                impl<'tcx> TyCtxt<'tcx> {
                    fn bar(self, arg1: Ty) -> RetTy {
                        (self.fn_provider.bar)((arg1,))
                    }
                }
                struct FnProvider {
                    bar: for<'tcx> fn(_: TyCtxt<'tcx>, _: (Ty1,)) -> RetTy,
                }
                impl FnProvider {
                    fn empty() -> Self {
                        FnProvider {
                            bar: |_| bug!("the function `bar` has not been initialized"),
                        }
                    }
                }
            },
        )
    }

    #[test]
    fn single_fn_and_two_arg() {
        assert_macro(
            early_declared_functions,
            quote! { fn foo(arg1: Ty1, arg2: Ty2) -> RetTy; },
            quote! {
                impl<'tcx> TyCtxt<'tcx> {
                    fn foo(self, arg1: Ty1, arg2: Ty2) -> RetTy {
                        (self.fn_provider.foo)((arg1, arg2,))
                    }
                }

                struct FnProvider {
                    foo: for<'tcx> fn(_: TyCtxt<'tcx>, _: (Ty1, Ty2,)) -> RetTy,
                }

                impl FnProvider {
                    fn empty() -> Self {
                        FnProvider {
                            foo: |_| bug!("the function `foo` has not been initialized"),
                        }
                    }
                }
            },
        )
    }

    #[test]
    fn multiple_fn() {
        assert_macro(
            early_declared_functions,
            quote! {
                fn foo(arg1: Ty1, arg2: Ty2) -> RetTy;

                fn bar(arg1: Ty) -> RetTy;
            },
            quote! {
                impl<'tcx> TyCtxt<'tcx> {
                    fn foo(self, arg1: Ty1, arg2: Ty2) -> RetTy {
                        (self.fn_provider.foo)((arg1, arg2,))
                    }

                    fn bar(self, arg1: Ty) -> RetTy {
                        (self.fn_provider.bar)((arg1,))
                    }
                }

                struct FnProvider {
                    foo: for<'tcx> fn(_: TyCtxt<'tcx>, _: (Ty1, Ty2,)) -> RetTy,
                    bar: for<'tcx> fn(_: TyCtxt<'tcx>, _: (Ty1,)) -> RetTy,
                }

                impl FnProvider {
                    fn empty() -> Self {
                        FnProvider {
                            foo: |_| bug!("the function `foo` has not been initialized"),
                            bar: |_| bug!("the function `bar` has not been initialized"),
                        }
                    }
                }
            },
        )
    }
}
