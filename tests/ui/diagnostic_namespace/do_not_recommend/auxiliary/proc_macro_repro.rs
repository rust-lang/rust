#![feature(proc_macro_quote)]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, quote};

#[proc_macro_attribute]
pub fn repro(_args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse input that looks like `fn f(arg: &mut Arg);`
    let mut input = input.into_iter();
    assert_eq!(input.next().unwrap().to_string(), "fn");
    assert_eq!(input.next().unwrap().to_string(), "f");
    let TokenTree::Group(group) = input.next().unwrap() else { unreachable!() };
    let mut input = group.stream().into_iter();
    assert_eq!(input.next().unwrap().to_string(), "arg");
    assert_eq!(input.next().unwrap().to_string(), ":");
    let arg: TokenStream = input.collect();

    // Emit output:
    quote! {
        const _: fn() = {
            #[diagnostic::on_unimplemented(
                message = "mutable reference to C++ type requires a pin -- use Pin<&mut Arg>",
                label = "use `Pin<&mut Arg>`"
            )]
            trait ReferenceToUnpin_Arg {
                fn check_unpin() {}
            }
            #[diagnostic::do_not_recommend]
            impl<
                'a,
                T: ?::core::marker::Sized + ::core::marker::Unpin,
            > ReferenceToUnpin_Arg for &'a mut T {}
            <$arg as ReferenceToUnpin_Arg>::check_unpin
        };
    }
}
