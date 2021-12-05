// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::TokenStream;

macro_rules! checker {
    ($attr_name:ident, $expected:literal) => {
        #[proc_macro_attribute]
        pub fn $attr_name(attr: TokenStream, input: TokenStream) -> TokenStream {
            assert_eq!(input.to_string(), $expected);
            TokenStream::new()
        }
    }
}

checker!(attr_extern, r#"extern "C" { fn ffi(#[a1] arg1 : i32, #[a2] ...) ; }"#);
checker!(attr_extern_cvar, r#"unsafe extern "C" fn cvar(arg1 : i32, #[a1] mut args : ...) {}"#);
checker!(attr_alias, "type Alias = fn(#[a1] u8, #[a2] ...) ;");
checker!(attr_free, "fn free(#[a1] arg1 : u8) { let lam = | #[a2] W(x), #[a3] y | () ; }");
checker!(attr_inherent_1, "fn inherent1(#[a1] self, #[a2] arg1 : u8) {}");
checker!(attr_inherent_2, "fn inherent2(#[a1] & self, #[a2] arg1 : u8) {}");
checker!(attr_inherent_3, "fn inherent3 < 'a > (#[a1] & 'a mut self, #[a2] arg1 : u8) {}");
checker!(attr_inherent_4, "fn inherent4 < 'a > (#[a1] self : Box < Self >, #[a2] arg1 : u8) {}");
checker!(attr_inherent_issue_64682, "fn inherent5(#[a1] #[a2] arg1 : u8, #[a3] arg2 : u8) {}");
checker!(attr_trait_1, "fn trait1(#[a1] self, #[a2] arg1 : u8) ;");
checker!(attr_trait_2, "fn trait2(#[a1] & self, #[a2] arg1 : u8) ;");
checker!(attr_trait_3, "fn trait3 < 'a > (#[a1] & 'a mut self, #[a2] arg1 : u8) ;");
checker!(attr_trait_4, r#"fn trait4 < 'a >
(#[a1] self : Box < Self >, #[a2] arg1 : u8, #[a3] Vec < u8 >) ;"#);
checker!(attr_trait_issue_64682, "fn trait5(#[a1] #[a2] arg1 : u8, #[a3] arg2 : u8) ;");
checker!(rename_params, r#"impl Foo
{
    fn hello(#[angery(true)] a : i32, #[a2] b : i32, #[what = "how"] c : u32)
    {} fn
    hello2(#[a1] #[a2] a : i32, #[what = "how"] b : i32, #[angery(true)] c :
           u32) {} fn
    hello_self(#[a1] #[a2] & self, #[a1] #[a2] a : i32, #[what = "how"] b :
               i32, #[angery(true)] c : u32) {}
}"#);
