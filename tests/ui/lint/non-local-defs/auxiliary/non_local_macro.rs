#[macro_export]
macro_rules! non_local_impl {
    ($a:ident) => {
        const _IMPL_DEBUG: () = {
            impl ::std::fmt::Debug for $a {
                fn fmt(&self, _: &mut ::std::fmt::Formatter<'_>)
                    -> ::std::result::Result<(), ::std::fmt::Error>
                {
                    todo!()
                }
            }
        };
    }
}

#[macro_export]
macro_rules! non_local_macro_rules {
    ($a:ident) => {
        const _MACRO_EXPORT: () = {
            #[macro_export]
            macro_rules! $a {
                () => {}
            }
        };
    }
}
