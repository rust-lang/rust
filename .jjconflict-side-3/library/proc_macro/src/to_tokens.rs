use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::rc::Rc;

use crate::{ConcatTreesHelper, Group, Ident, Literal, Punct, Span, TokenStream, TokenTree};

/// Types that can be interpolated inside a [`quote!`] invocation.
///
/// [`quote!`]: crate::quote!
#[unstable(feature = "proc_macro_totokens", issue = "130977")]
pub trait ToTokens {
    /// Write `self` to the given `TokenStream`.
    ///
    /// # Example
    ///
    /// Example implementation for a struct representing Rust paths like
    /// `std::cmp::PartialEq`:
    ///
    /// ```
    /// #![feature(proc_macro_totokens)]
    ///
    /// use std::iter;
    /// use proc_macro::{Spacing, Punct, TokenStream, TokenTree, ToTokens};
    ///
    /// pub struct Path {
    ///     pub global: bool,
    ///     pub segments: Vec<PathSegment>,
    /// }
    ///
    /// impl ToTokens for Path {
    ///     fn to_tokens(&self, tokens: &mut TokenStream) {
    ///         for (i, segment) in self.segments.iter().enumerate() {
    ///             if i > 0 || self.global {
    ///                 // Double colon `::`
    ///                 tokens.extend(iter::once(TokenTree::from(Punct::new(':', Spacing::Joint))));
    ///                 tokens.extend(iter::once(TokenTree::from(Punct::new(':', Spacing::Alone))));
    ///             }
    ///             segment.to_tokens(tokens);
    ///         }
    ///     }
    /// }
    /// #
    /// # pub struct PathSegment;
    /// #
    /// # impl ToTokens for PathSegment {
    /// #     fn to_tokens(&self, tokens: &mut TokenStream) {
    /// #         unimplemented!()
    /// #     }
    /// # }
    /// ```
    fn to_tokens(&self, tokens: &mut TokenStream);

    /// Convert `self` directly into a `TokenStream` object.
    ///
    /// This method is implicitly implemented using `to_tokens`, and acts as a
    /// convenience method for consumers of the `ToTokens` trait.
    fn to_token_stream(&self) -> TokenStream {
        let mut tokens = TokenStream::new();
        self.to_tokens(&mut tokens);
        tokens
    }

    /// Convert `self` directly into a `TokenStream` object.
    ///
    /// This method is implicitly implemented using `to_tokens`, and acts as a
    /// convenience method for consumers of the `ToTokens` trait.
    fn into_token_stream(self) -> TokenStream
    where
        Self: Sized,
    {
        self.to_token_stream()
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for TokenTree {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend_one(self.clone());
    }

    fn into_token_stream(self) -> TokenStream {
        let mut builder = ConcatTreesHelper::new(1);
        builder.push(self);
        builder.build()
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for TokenStream {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(self.clone());
    }

    fn into_token_stream(self) -> TokenStream {
        self
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for Literal {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend_one(TokenTree::from(self.clone()));
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for Ident {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend_one(TokenTree::from(self.clone()));
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for Punct {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend_one(TokenTree::from(self.clone()));
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for Group {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend_one(TokenTree::from(self.clone()));
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens + ?Sized> ToTokens for &T {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens + ?Sized> ToTokens for &mut T {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens + ?Sized> ToTokens for Box<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens + ?Sized> ToTokens for Rc<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens + ToOwned + ?Sized> ToTokens for Cow<'_, T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        (**self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl<T: ToTokens> ToTokens for Option<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Some(t) = self {
            t.to_tokens(tokens);
        }
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for u8 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::u8_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for u16 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::u16_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for u32 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::u32_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for u64 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::u64_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for u128 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::u128_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for i8 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::i8_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for i16 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::i16_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for i32 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::i32_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for i64 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::i64_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for i128 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::i128_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for f32 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::f32_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for f64 {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::f64_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for usize {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::usize_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for isize {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::isize_suffixed(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for bool {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let word = if *self { "true" } else { "false" };
        Ident::new(word, Span::call_site()).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for char {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::character(*self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for str {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::string(self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for String {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::string(self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for CStr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::c_string(self).to_tokens(tokens)
    }
}

#[unstable(feature = "proc_macro_totokens", issue = "130977")]
impl ToTokens for CString {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        Literal::c_string(self).to_tokens(tokens)
    }
}
