extern crate proc_macro;
use proc_macro::{Literal, TokenStream};

#[proc_macro]
pub fn check(_: TokenStream) -> TokenStream {
    // In the following test cases, the `---` may look like the start of frontmatter but it is not!
    // That's because it would be backward incompatible to interpret them as such in the latest
    // stable edition. That's not only the case due to the feature gate error but also due to the
    // fact that we "eagerly" emit errors on malformed frontmatter.

    // issue: <https://github.com/rust-lang/rust/issues/145520>
    _ = "---".parse::<TokenStream>();
    // Just a sequence of regular Rust punctuation tokens.
    assert_eq!(6, "---\n---".parse::<TokenStream>().unwrap().into_iter().count());

    // issue: <https://github.com/rust-lang/rust/issues/146132>
    assert!("---".parse::<Literal>().is_err());

    Default::default()
}
