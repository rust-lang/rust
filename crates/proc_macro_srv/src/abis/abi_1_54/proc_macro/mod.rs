//! lib-proc-macro main module
//!
//! Copy from <https://github.com/rust-lang/rust/blob/6050e523bae6de61de4e060facc43dc512adaccd/src/libproc_macro/lib.rs>
//! augmented with removing unstable features

// NOTE(@edwin0cheng):
// Because we just copy the bridge module from rustc for ABI compatible
// There are some unused stuffs inside it.
// We suppress these warning here.
#[doc(hidden)]
#[allow(unused_macros)]
#[allow(unused_variables)]
pub mod bridge;

mod diagnostic;

pub use diagnostic::{Diagnostic, Level, MultiSpan};

use std::ops::{Bound, RangeBounds};
use std::path::PathBuf;
use std::str::FromStr;
use std::{fmt, iter, mem};

/// The main type provided by this crate, representing an abstract stream of
/// tokens, or, more specifically, a sequence of token trees.
/// The type provide interfaces for iterating over those token trees and, conversely,
/// collecting a number of token trees into one stream.
///
/// This is both the input and output of `#[proc_macro]`, `#[proc_macro_attribute]`
/// and `#[proc_macro_derive]` definitions.
#[derive(Clone)]
pub struct TokenStream(bridge::client::TokenStream);

/// Error returned from `TokenStream::from_str`
#[derive(Debug)]
pub struct LexError {
    _inner: (),
}
impl LexError {
    fn new() -> Self {
        LexError { _inner: () }
    }
}

impl TokenStream {
    /// Returns an empty `TokenStream` containing no token trees.
    pub fn new() -> TokenStream {
        TokenStream(bridge::client::TokenStream::new())
    }

    /// Checks if this `TokenStream` is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Attempts to break the string into tokens and parse those tokens into a token stream.
/// May fail for a number of reasons, for example, if the string contains unbalanced delimiters
/// or characters not existing in the language.
/// All tokens in the parsed stream get `Span::call_site()` spans.
///
/// NOTE: some errors may cause panics instead of returning `LexError`. We reserve the right to
/// change these errors into `LexError`s later.
impl FromStr for TokenStream {
    type Err = LexError;

    fn from_str(src: &str) -> Result<TokenStream, LexError> {
        Ok(TokenStream(bridge::client::TokenStream::from_str(src)))
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for TokenStream {
//     fn to_string(&self) -> String {
//         self.0.to_string()
//     }
// }

/// Prints the token stream as a string that is supposed to be losslessly convertible back
/// into the same token stream (modulo spans), except for possibly `TokenTree::Group`s
/// with `Delimiter::None` delimiters and negative numeric literals.
impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

/// Prints token in a form convenient for debugging.
impl fmt::Debug for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenStream ")?;
        f.debug_list().entries(self.clone()).finish()
    }
}

/// Creates a token stream containing a single token tree.
impl From<TokenTree> for TokenStream {
    fn from(tree: TokenTree) -> TokenStream {
        TokenStream(bridge::client::TokenStream::from_token_tree(match tree {
            TokenTree::Group(tt) => bridge::TokenTree::Group(tt.0),
            TokenTree::Punct(tt) => bridge::TokenTree::Punct(tt.0),
            TokenTree::Ident(tt) => bridge::TokenTree::Ident(tt.0),
            TokenTree::Literal(tt) => bridge::TokenTree::Literal(tt.0),
        }))
    }
}

/// Collects a number of token trees into a single stream.
impl iter::FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(trees: I) -> Self {
        trees.into_iter().map(TokenStream::from).collect()
    }
}

/// A "flattening" operation on token streams, collects token trees
/// from multiple token streams into a single stream.
impl iter::FromIterator<TokenStream> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenStream>>(streams: I) -> Self {
        let mut builder = bridge::client::TokenStreamBuilder::new();
        streams.into_iter().for_each(|stream| builder.push(stream.0));
        TokenStream(builder.build())
    }
}

impl Extend<TokenTree> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenTree>>(&mut self, trees: I) {
        self.extend(trees.into_iter().map(TokenStream::from));
    }
}

impl Extend<TokenStream> for TokenStream {
    fn extend<I: IntoIterator<Item = TokenStream>>(&mut self, streams: I) {
        // FIXME(eddyb) Use an optimized implementation if/when possible.
        *self = iter::once(mem::replace(self, Self::new())).chain(streams).collect();
    }
}

/// Public implementation details for the `TokenStream` type, such as iterators.
pub mod token_stream {
    use super::{bridge, Group, Ident, Literal, Punct, TokenStream, TokenTree};

    /// An iterator over `TokenStream`'s `TokenTree`s.
    /// The iteration is "shallow", e.g., the iterator doesn't recurse into delimited groups,
    /// and returns whole groups as token trees.
    #[derive(Clone)]
    pub struct IntoIter(bridge::client::TokenStreamIter);

    impl Iterator for IntoIter {
        type Item = TokenTree;

        fn next(&mut self) -> Option<TokenTree> {
            self.0.next().map(|tree| match tree {
                bridge::TokenTree::Group(tt) => TokenTree::Group(Group(tt)),
                bridge::TokenTree::Punct(tt) => TokenTree::Punct(Punct(tt)),
                bridge::TokenTree::Ident(tt) => TokenTree::Ident(Ident(tt)),
                bridge::TokenTree::Literal(tt) => TokenTree::Literal(Literal(tt)),
            })
        }
    }

    impl IntoIterator for TokenStream {
        type Item = TokenTree;
        type IntoIter = IntoIter;

        fn into_iter(self) -> IntoIter {
            IntoIter(self.0.into_iter())
        }
    }
}

/// A region of source code, along with macro expansion information.
#[derive(Copy, Clone)]
pub struct Span(bridge::client::Span);

macro_rules! diagnostic_method {
    ($name:ident, $level:expr) => {
        /// Creates a new `Diagnostic` with the given `message` at the span
        /// `self`.
        pub fn $name<T: Into<String>>(self, message: T) -> Diagnostic {
            Diagnostic::spanned(self, $level, message)
        }
    };
}

impl Span {
    /// A span that resolves at the macro definition site.
    pub fn def_site() -> Span {
        Span(bridge::client::Span::def_site())
    }

    /// The span of the invocation of the current procedural macro.
    /// Identifiers created with this span will be resolved as if they were written
    /// directly at the macro call location (call-site hygiene) and other code
    /// at the macro call site will be able to refer to them as well.
    pub fn call_site() -> Span {
        Span(bridge::client::Span::call_site())
    }

    /// A span that represents `macro_rules` hygiene, and sometimes resolves at the macro
    /// definition site (local variables, labels, `$crate`) and sometimes at the macro
    /// call site (everything else).
    /// The span location is taken from the call-site.
    pub fn mixed_site() -> Span {
        Span(bridge::client::Span::mixed_site())
    }

    /// The original source file into which this span points.
    pub fn source_file(&self) -> SourceFile {
        SourceFile(self.0.source_file())
    }

    /// The `Span` for the tokens in the previous macro expansion from which
    /// `self` was generated from, if any.
    pub fn parent(&self) -> Option<Span> {
        self.0.parent().map(Span)
    }

    /// The span for the origin source code that `self` was generated from. If
    /// this `Span` wasn't generated from other macro expansions then the return
    /// value is the same as `*self`.
    pub fn source(&self) -> Span {
        Span(self.0.source())
    }

    /// Gets the starting line/column in the source file for this span.
    pub fn start(&self) -> LineColumn {
        self.0.start()
    }

    /// Gets the ending line/column in the source file for this span.
    pub fn end(&self) -> LineColumn {
        self.0.end()
    }

    /// Creates a new span encompassing `self` and `other`.
    ///
    /// Returns `None` if `self` and `other` are from different files.
    pub fn join(&self, other: Span) -> Option<Span> {
        self.0.join(other.0).map(Span)
    }

    /// Creates a new span with the same line/column information as `self` but
    /// that resolves symbols as though it were at `other`.
    pub fn resolved_at(&self, other: Span) -> Span {
        Span(self.0.resolved_at(other.0))
    }

    /// Creates a new span with the same name resolution behavior as `self` but
    /// with the line/column information of `other`.
    pub fn located_at(&self, other: Span) -> Span {
        other.resolved_at(*self)
    }

    /// Compares to spans to see if they're equal.
    pub fn eq(&self, other: &Span) -> bool {
        self.0 == other.0
    }

    /// Returns the source text behind a span. This preserves the original source
    /// code, including spaces and comments. It only returns a result if the span
    /// corresponds to real source code.
    ///
    /// Note: The observable result of a macro should only rely on the tokens and
    /// not on this source text. The result of this function is a best effort to
    /// be used for diagnostics only.
    pub fn source_text(&self) -> Option<String> {
        self.0.source_text()
    }

    diagnostic_method!(error, Level::Error);
    diagnostic_method!(warning, Level::Warning);
    diagnostic_method!(note, Level::Note);
    diagnostic_method!(help, Level::Help);
}

/// Prints a span in a form convenient for debugging.
impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A line-column pair representing the start or end of a `Span`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LineColumn {
    /// The 1-indexed line in the source file on which the span starts or ends (inclusive).
    pub line: usize,
    /// The 0-indexed column (in UTF-8 characters) in the source file on which
    /// the span starts or ends (inclusive).
    pub column: usize,
}

/// The source file of a given `Span`.
#[derive(Clone)]
pub struct SourceFile(bridge::client::SourceFile);

impl SourceFile {
    /// Gets the path to this source file.
    ///
    /// ### Note
    /// If the code span associated with this `SourceFile` was generated by an external macro, this
    /// macro, this may not be an actual path on the filesystem. Use [`is_real`] to check.
    ///
    /// Also note that even if `is_real` returns `true`, if `--remap-path-prefix` was passed on
    /// the command line, the path as given may not actually be valid.
    ///
    /// [`is_real`]: #method.is_real
    pub fn path(&self) -> PathBuf {
        PathBuf::from(self.0.path())
    }

    /// Returns `true` if this source file is a real source file, and not generated by an external
    /// macro's expansion.
    pub fn is_real(&self) -> bool {
        // This is a hack until intercrate spans are implemented and we can have real source files
        // for spans generated in external macros.
        // https://github.com/rust-lang/rust/pull/43604#issuecomment-333334368
        self.0.is_real()
    }
}

impl fmt::Debug for SourceFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SourceFile")
            .field("path", &self.path())
            .field("is_real", &self.is_real())
            .finish()
    }
}

impl PartialEq for SourceFile {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for SourceFile {}

/// A single token or a delimited sequence of token trees (e.g., `[1, (), ..]`).
#[derive(Clone)]
pub enum TokenTree {
    /// A token stream surrounded by bracket delimiters.
    Group(Group),
    /// An identifier.
    Ident(Ident),
    /// A single punctuation character (`+`, `,`, `$`, etc.).
    Punct(Punct),
    /// A literal character (`'a'`), string (`"hello"`), number (`2.3`), etc.
    Literal(Literal),
}

impl TokenTree {
    /// Returns the span of this tree, delegating to the `span` method of
    /// the contained token or a delimited stream.
    pub fn span(&self) -> Span {
        match *self {
            TokenTree::Group(ref t) => t.span(),
            TokenTree::Ident(ref t) => t.span(),
            TokenTree::Punct(ref t) => t.span(),
            TokenTree::Literal(ref t) => t.span(),
        }
    }

    /// Configures the span for *only this token*.
    ///
    /// Note that if this token is a `Group` then this method will not configure
    /// the span of each of the internal tokens, this will simply delegate to
    /// the `set_span` method of each variant.
    pub fn set_span(&mut self, span: Span) {
        match *self {
            TokenTree::Group(ref mut t) => t.set_span(span),
            TokenTree::Ident(ref mut t) => t.set_span(span),
            TokenTree::Punct(ref mut t) => t.set_span(span),
            TokenTree::Literal(ref mut t) => t.set_span(span),
        }
    }
}

/// Prints token tree in a form convenient for debugging.
impl fmt::Debug for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Each of these has the name in the struct type in the derived debug,
        // so don't bother with an extra layer of indirection
        match *self {
            TokenTree::Group(ref tt) => tt.fmt(f),
            TokenTree::Ident(ref tt) => tt.fmt(f),
            TokenTree::Punct(ref tt) => tt.fmt(f),
            TokenTree::Literal(ref tt) => tt.fmt(f),
        }
    }
}

impl From<Group> for TokenTree {
    fn from(g: Group) -> TokenTree {
        TokenTree::Group(g)
    }
}

impl From<Ident> for TokenTree {
    fn from(g: Ident) -> TokenTree {
        TokenTree::Ident(g)
    }
}

impl From<Punct> for TokenTree {
    fn from(g: Punct) -> TokenTree {
        TokenTree::Punct(g)
    }
}

impl From<Literal> for TokenTree {
    fn from(g: Literal) -> TokenTree {
        TokenTree::Literal(g)
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for TokenTree {
//     fn to_string(&self) -> String {
//         match *self {
//             TokenTree::Group(ref t) => t.to_string(),
//             TokenTree::Ident(ref t) => t.to_string(),
//             TokenTree::Punct(ref t) => t.to_string(),
//             TokenTree::Literal(ref t) => t.to_string(),
//         }
//     }
// }

/// Prints the token tree as a string that is supposed to be losslessly convertible back
/// into the same token tree (modulo spans), except for possibly `TokenTree::Group`s
/// with `Delimiter::None` delimiters and negative numeric literals.
impl fmt::Display for TokenTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

/// A delimited token stream.
///
/// A `Group` internally contains a `TokenStream` which is surrounded by `Delimiter`s.
#[derive(Clone)]
pub struct Group(bridge::client::Group);

/// Describes how a sequence of token trees is delimited.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Delimiter {
    /// `( ... )`
    Parenthesis,
    /// `{ ... }`
    Brace,
    /// `[ ... ]`
    Bracket,
    /// `Ø ... Ø`
    /// An implicit delimiter, that may, for example, appear around tokens coming from a
    /// "macro variable" `$var`. It is important to preserve operator priorities in cases like
    /// `$var * 3` where `$var` is `1 + 2`.
    /// Implicit delimiters may not survive roundtrip of a token stream through a string.
    None,
}

impl Group {
    /// Creates a new `Group` with the given delimiter and token stream.
    ///
    /// This constructor will set the span for this group to
    /// `Span::call_site()`. To change the span you can use the `set_span`
    /// method below.
    pub fn new(delimiter: Delimiter, stream: TokenStream) -> Group {
        Group(bridge::client::Group::new(delimiter, stream.0))
    }

    /// Returns the delimiter of this `Group`
    pub fn delimiter(&self) -> Delimiter {
        self.0.delimiter()
    }

    /// Returns the `TokenStream` of tokens that are delimited in this `Group`.
    ///
    /// Note that the returned token stream does not include the delimiter
    /// returned above.
    pub fn stream(&self) -> TokenStream {
        TokenStream(self.0.stream())
    }

    /// Returns the span for the delimiters of this token stream, spanning the
    /// entire `Group`.
    ///
    /// ```text
    /// pub fn span(&self) -> Span {
    ///            ^^^^^^^
    /// ```
    pub fn span(&self) -> Span {
        Span(self.0.span())
    }

    /// Returns the span pointing to the opening delimiter of this group.
    ///
    /// ```text
    /// pub fn span_open(&self) -> Span {
    ///                 ^
    /// ```
    pub fn span_open(&self) -> Span {
        Span(self.0.span_open())
    }

    /// Returns the span pointing to the closing delimiter of this group.
    ///
    /// ```text
    /// pub fn span_close(&self) -> Span {
    ///                        ^
    /// ```
    pub fn span_close(&self) -> Span {
        Span(self.0.span_close())
    }

    /// Configures the span for this `Group`'s delimiters, but not its internal
    /// tokens.
    ///
    /// This method will **not** set the span of all the internal tokens spanned
    /// by this group, but rather it will only set the span of the delimiter
    /// tokens at the level of the `Group`.
    pub fn set_span(&mut self, span: Span) {
        self.0.set_span(span.0);
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for Group {
//     fn to_string(&self) -> String {
//         TokenStream::from(TokenTree::from(self.clone())).to_string()
//     }
// }

/// Prints the group as a string that should be losslessly convertible back
/// into the same group (modulo spans), except for possibly `TokenTree::Group`s
/// with `Delimiter::None` delimiters.
impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl fmt::Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Group")
            .field("delimiter", &self.delimiter())
            .field("stream", &self.stream())
            .field("span", &self.span())
            .finish()
    }
}

/// An `Punct` is an single punctuation character like `+`, `-` or `#`.
///
/// Multi-character operators like `+=` are represented as two instances of `Punct` with different
/// forms of `Spacing` returned.
#[derive(Clone)]
pub struct Punct(bridge::client::Punct);

/// Whether an `Punct` is followed immediately by another `Punct` or
/// followed by another token or whitespace.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Spacing {
    /// e.g., `+` is `Alone` in `+ =`, `+ident` or `+()`.
    Alone,
    /// e.g., `+` is `Joint` in `+=` or `'#`.
    /// Additionally, single quote `'` can join with identifiers to form lifetimes `'ident`.
    Joint,
}

impl Punct {
    /// Creates a new `Punct` from the given character and spacing.
    /// The `ch` argument must be a valid punctuation character permitted by the language,
    /// otherwise the function will panic.
    ///
    /// The returned `Punct` will have the default span of `Span::call_site()`
    /// which can be further configured with the `set_span` method below.
    pub fn new(ch: char, spacing: Spacing) -> Punct {
        Punct(bridge::client::Punct::new(ch, spacing))
    }

    /// Returns the value of this punctuation character as `char`.
    pub fn as_char(&self) -> char {
        self.0.as_char()
    }

    /// Returns the spacing of this punctuation character, indicating whether it's immediately
    /// followed by another `Punct` in the token stream, so they can potentially be combined into
    /// a multi-character operator (`Joint`), or it's followed by some other token or whitespace
    /// (`Alone`) so the operator has certainly ended.
    pub fn spacing(&self) -> Spacing {
        self.0.spacing()
    }

    /// Returns the span for this punctuation character.
    pub fn span(&self) -> Span {
        Span(self.0.span())
    }

    /// Configure the span for this punctuation character.
    pub fn set_span(&mut self, span: Span) {
        self.0 = self.0.with_span(span.0);
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for Punct {
//     fn to_string(&self) -> String {
//         TokenStream::from(TokenTree::from(self.clone())).to_string()
//     }
// }

/// Prints the punctuation character as a string that should be losslessly convertible
/// back into the same character.
impl fmt::Display for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl fmt::Debug for Punct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Punct")
            .field("ch", &self.as_char())
            .field("spacing", &self.spacing())
            .field("span", &self.span())
            .finish()
    }
}

/// An identifier (`ident`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Ident(bridge::client::Ident);

impl Ident {
    /// Creates a new `Ident` with the given `string` as well as the specified
    /// `span`.
    /// The `string` argument must be a valid identifier permitted by the
    /// language, otherwise the function will panic.
    ///
    /// Note that `span`, currently in rustc, configures the hygiene information
    /// for this identifier.
    ///
    /// As of this time `Span::call_site()` explicitly opts-in to "call-site" hygiene
    /// meaning that identifiers created with this span will be resolved as if they were written
    /// directly at the location of the macro call, and other code at the macro call site will be
    /// able to refer to them as well.
    ///
    /// Later spans like `Span::def_site()` will allow to opt-in to "definition-site" hygiene
    /// meaning that identifiers created with this span will be resolved at the location of the
    /// macro definition and other code at the macro call site will not be able to refer to them.
    ///
    /// Due to the current importance of hygiene this constructor, unlike other
    /// tokens, requires a `Span` to be specified at construction.
    pub fn new(string: &str, span: Span) -> Ident {
        Ident(bridge::client::Ident::new(string, span.0, false))
    }

    /// Same as `Ident::new`, but creates a raw identifier (`r#ident`).
    pub fn new_raw(string: &str, span: Span) -> Ident {
        Ident(bridge::client::Ident::new(string, span.0, true))
    }

    /// Returns the span of this `Ident`, encompassing the entire string returned
    /// by `as_str`.
    pub fn span(&self) -> Span {
        Span(self.0.span())
    }

    /// Configures the span of this `Ident`, possibly changing its hygiene context.
    pub fn set_span(&mut self, span: Span) {
        self.0 = self.0.with_span(span.0);
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for Ident {
//     fn to_string(&self) -> String {
//         TokenStream::from(TokenTree::from(self.clone())).to_string()
//     }
// }

/// Prints the identifier as a string that should be losslessly convertible
/// back into the same identifier.
impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ident")
            .field("ident", &self.to_string())
            .field("span", &self.span())
            .finish()
    }
}

/// A literal string (`"hello"`), byte string (`b"hello"`),
/// character (`'a'`), byte character (`b'a'`), an integer or floating point number
/// with or without a suffix (`1`, `1u8`, `2.3`, `2.3f32`).
/// Boolean literals like `true` and `false` do not belong here, they are `Ident`s.
#[derive(Clone)]
pub struct Literal(bridge::client::Literal);

macro_rules! suffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new suffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1u32` where the integer
        /// value specified is the first part of the token and the integral is
        /// also suffixed at the end.
        /// Literals created from negative numbers may not survive round-trips through
        /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        pub fn $name(n: $kind) -> Literal {
            Literal(bridge::client::Literal::typed_integer(&n.to_string(), stringify!($kind)))
        }
    )*)
}

macro_rules! unsuffixed_int_literals {
    ($($name:ident => $kind:ident,)*) => ($(
        /// Creates a new unsuffixed integer literal with the specified value.
        ///
        /// This function will create an integer like `1` where the integer
        /// value specified is the first part of the token. No suffix is
        /// specified on this token, meaning that invocations like
        /// `Literal::i8_unsuffixed(1)` are equivalent to
        /// `Literal::u32_unsuffixed(1)`.
        /// Literals created from negative numbers may not survive rountrips through
        /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
        ///
        /// Literals created through this method have the `Span::call_site()`
        /// span by default, which can be configured with the `set_span` method
        /// below.
        pub fn $name(n: $kind) -> Literal {
            Literal(bridge::client::Literal::integer(&n.to_string()))
        }
    )*)
}

impl Literal {
    suffixed_int_literals! {
        u8_suffixed => u8,
        u16_suffixed => u16,
        u32_suffixed => u32,
        u64_suffixed => u64,
        u128_suffixed => u128,
        usize_suffixed => usize,
        i8_suffixed => i8,
        i16_suffixed => i16,
        i32_suffixed => i32,
        i64_suffixed => i64,
        i128_suffixed => i128,
        isize_suffixed => isize,
    }

    unsuffixed_int_literals! {
        u8_unsuffixed => u8,
        u16_unsuffixed => u16,
        u32_unsuffixed => u32,
        u64_unsuffixed => u64,
        u128_unsuffixed => u128,
        usize_unsuffixed => usize,
        i8_unsuffixed => i8,
        i16_unsuffixed => i16,
        i32_unsuffixed => i32,
        i64_unsuffixed => i64,
        i128_unsuffixed => i128,
        isize_unsuffixed => isize,
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    /// Literals created from negative numbers may not survive rountrips through
    /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    pub fn f32_unsuffixed(n: f32) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal(bridge::client::Literal::float(&n.to_string()))
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This constructor will create a literal like `1.0f32` where the value
    /// specified is the preceding part of the token and `f32` is the suffix of
    /// the token. This token will always be inferred to be an `f32` in the
    /// compiler.
    /// Literals created from negative numbers may not survive rountrips through
    /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    pub fn f32_suffixed(n: f32) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal(bridge::client::Literal::f32(&n.to_string()))
    }

    /// Creates a new unsuffixed floating-point literal.
    ///
    /// This constructor is similar to those like `Literal::i8_unsuffixed` where
    /// the float's value is emitted directly into the token but no suffix is
    /// used, so it may be inferred to be a `f64` later in the compiler.
    /// Literals created from negative numbers may not survive rountrips through
    /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    pub fn f64_unsuffixed(n: f64) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal(bridge::client::Literal::float(&n.to_string()))
    }

    /// Creates a new suffixed floating-point literal.
    ///
    /// This constructor will create a literal like `1.0f64` where the value
    /// specified is the preceding part of the token and `f64` is the suffix of
    /// the token. This token will always be inferred to be an `f64` in the
    /// compiler.
    /// Literals created from negative numbers may not survive rountrips through
    /// `TokenStream` or strings and may be broken into two tokens (`-` and positive literal).
    ///
    /// # Panics
    ///
    /// This function requires that the specified float is finite, for
    /// example if it is infinity or NaN this function will panic.
    pub fn f64_suffixed(n: f64) -> Literal {
        if !n.is_finite() {
            panic!("Invalid float literal {}", n);
        }
        Literal(bridge::client::Literal::f64(&n.to_string()))
    }

    /// String literal.
    pub fn string(string: &str) -> Literal {
        Literal(bridge::client::Literal::string(string))
    }

    /// Character literal.
    pub fn character(ch: char) -> Literal {
        Literal(bridge::client::Literal::character(ch))
    }

    /// Byte string literal.
    pub fn byte_string(bytes: &[u8]) -> Literal {
        Literal(bridge::client::Literal::byte_string(bytes))
    }

    /// Returns the span encompassing this literal.
    pub fn span(&self) -> Span {
        Span(self.0.span())
    }

    /// Configures the span associated for this literal.
    pub fn set_span(&mut self, span: Span) {
        self.0.set_span(span.0);
    }

    /// Returns a `Span` that is a subset of `self.span()` containing only the
    /// source bytes in range `range`. Returns `None` if the would-be trimmed
    /// span is outside the bounds of `self`.
    // FIXME(SergioBenitez): check that the byte range starts and ends at a
    // UTF-8 boundary of the source. otherwise, it's likely that a panic will
    // occur elsewhere when the source text is printed.
    // FIXME(SergioBenitez): there is no way for the user to know what
    // `self.span()` actually maps to, so this method can currently only be
    // called blindly. For example, `to_string()` for the character 'c' returns
    // "'\u{63}'"; there is no way for the user to know whether the source text
    // was 'c' or whether it was '\u{63}'.
    pub fn subspan<R: RangeBounds<usize>>(&self, range: R) -> Option<Span> {
        // HACK(eddyb) something akin to `Option::cloned`, but for `Bound<&T>`.
        fn cloned_bound<T: Clone>(bound: Bound<&T>) -> Bound<T> {
            match bound {
                Bound::Included(x) => Bound::Included(x.clone()),
                Bound::Excluded(x) => Bound::Excluded(x.clone()),
                Bound::Unbounded => Bound::Unbounded,
            }
        }

        self.0.subspan(cloned_bound(range.start_bound()), cloned_bound(range.end_bound())).map(Span)
    }
}

// N.B., the bridge only provides `to_string`, implement `fmt::Display`
// based on it (the reverse of the usual relationship between the two).
// impl ToString for Literal {
//     fn to_string(&self) -> String {
//         TokenStream::from(TokenTree::from(self.clone())).to_string()
//     }
// }

/// Prints the literal as a string that should be losslessly convertible
/// back into the same literal (except for possible rounding for floating point literals).
impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string())
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // FIXME(eddyb) `Literal` should not expose internal `Debug` impls.
        self.0.fmt(f)
    }
}

impl FromStr for Literal {
    type Err = LexError;

    fn from_str(src: &str) -> Result<Self, LexError> {
        match bridge::client::Literal::from_str(src) {
            Ok(literal) => Ok(Literal(literal)),
            Err(()) => Err(LexError::new()),
        }
    }
}

pub mod tracked_env {
    use std::env::{self, VarError};
    use std::ffi::OsStr;

    /// Retrieve an environment variable and add it to build dependency info.
    /// Build system executing the compiler will know that the variable was accessed during
    /// compilation, and will be able to rerun the build when the value of that variable changes.
    /// Besides the dependency tracking this function should be equivalent to `env::var` from the
    /// standard library, except that the argument must be UTF-8.
    pub fn var<K: AsRef<OsStr> + AsRef<str>>(key: K) -> Result<String, VarError> {
        use std::ops::Deref;

        let key: &str = key.as_ref();
        let value = env::var(key);
        super::bridge::client::FreeFunctions::track_env_var(
            key,
            value.as_ref().map(|t| t.deref()).ok(),
        );
        value
    }
}
