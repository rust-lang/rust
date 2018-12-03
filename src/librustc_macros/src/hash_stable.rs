use synstructure;
use syn;
use proc_macro2;

pub fn hash_stable_derive(mut s: synstructure::Structure) -> proc_macro2::TokenStream {
    let generic: syn::GenericParam = parse_quote!('__ctx);
    s.add_bounds(synstructure::AddBounds::Generics);
    s.add_impl_generic(generic);
    let body = s.each(|bi| quote!{
        ::rustc_data_structures::stable_hasher::HashStable::hash_stable(#bi, __hcx, __hasher);
    });

    let discriminant = match s.ast().data {
        syn::Data::Enum(_) => quote! {
            ::std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        },
        syn::Data::Struct(_) => quote! {},
        syn::Data::Union(_) => panic!("cannot derive on union"),
    };

    s.bound_impl(quote!(::rustc_data_structures::stable_hasher::HashStable
                        <::rustc::ich::StableHashingContext<'__ctx>>), quote!{
        fn hash_stable<__W: ::rustc_data_structures::stable_hasher::StableHasherResult>(
            &self,
            __hcx: &mut ::rustc::ich::StableHashingContext<'__ctx>,
            __hasher: &mut ::rustc_data_structures::stable_hasher::StableHasher<__W>) {
            #discriminant
            match *self { #body }
        }
    })
}
