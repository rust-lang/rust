// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e., given a value, one can easily find the
//! type, and vice versa.

use arena::DroplessArena;
use rustc_data_structures::fx::FxHashMap;
use serialize::{Decodable, Decoder, Encodable, Encoder};

use std::fmt;
use std::str;
use std::cmp::{PartialEq, Ordering, PartialOrd, Ord};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;

use hygiene::SyntaxContext;
use {Span, DUMMY_SP, GLOBALS};

#[derive(Copy, Clone, Eq)]
pub struct Ident {
    pub name: Symbol,
    pub span: Span,
}

impl Ident {
    #[inline]
    pub const fn new(name: Symbol, span: Span) -> Ident {
        Ident { name, span }
    }

    #[inline]
    pub const fn with_empty_ctxt(name: Symbol) -> Ident {
        Ident::new(name, DUMMY_SP)
    }

    /// Maps an interned string to an identifier with an empty syntax context.
    pub fn from_interned_str(string: InternedString) -> Ident {
        Ident::with_empty_ctxt(string.as_symbol())
    }

    /// Maps a string to an identifier with an empty syntax context.
    pub fn from_str(string: &str) -> Ident {
        Ident::with_empty_ctxt(Symbol::intern(string))
    }

    /// Replace `lo` and `hi` with those from `span`, but keep hygiene context.
    pub fn with_span_pos(self, span: Span) -> Ident {
        Ident::new(self.name, span.with_ctxt(self.span.ctxt()))
    }

    pub fn without_first_quote(self) -> Ident {
        Ident::new(Symbol::intern(self.as_str().trim_left_matches('\'')), self.span)
    }

    /// "Normalize" ident for use in comparisons using "item hygiene".
    /// Identifiers with same string value become same if they came from the same "modern" macro
    /// (e.g., `macro` item, but not `macro_rules` item) and stay different if they came from
    /// different "modern" macros.
    /// Technically, this operation strips all non-opaque marks from ident's syntactic context.
    pub fn modern(self) -> Ident {
        Ident::new(self.name, self.span.modern())
    }

    /// "Normalize" ident for use in comparisons using "local variable hygiene".
    /// Identifiers with same string value become same if they came from the same non-transparent
    /// macro (e.g., `macro` or `macro_rules!` items) and stay different if they came from different
    /// non-transparent macros.
    /// Technically, this operation strips all transparent marks from ident's syntactic context.
    pub fn modern_and_legacy(self) -> Ident {
        Ident::new(self.name, self.span.modern_and_legacy())
    }

    pub fn gensym(self) -> Ident {
        Ident::new(self.name.gensymed(), self.span)
    }

    pub fn gensym_if_underscore(self) -> Ident {
        if self.name == keywords::Underscore.name() { self.gensym() } else { self }
    }

    pub fn as_str(self) -> LocalInternedString {
        self.name.as_str()
    }

    pub fn as_interned_str(self) -> InternedString {
        self.name.as_interned_str()
    }
}

impl PartialEq for Ident {
    fn eq(&self, rhs: &Self) -> bool {
        self.name == rhs.name && self.span.ctxt() == rhs.span.ctxt()
    }
}

impl Hash for Ident {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.span.ctxt().hash(state);
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{:?}", self.name, self.span.ctxt())
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl Encodable for Ident {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        if self.span.ctxt().modern() == SyntaxContext::empty() {
            s.emit_str(&self.as_str())
        } else { // FIXME(jseyfried): intercrate hygiene
            let mut string = "#".to_owned();
            string.push_str(&self.as_str());
            s.emit_str(&string)
        }
    }
}

impl Decodable for Ident {
    fn decode<D: Decoder>(d: &mut D) -> Result<Ident, D::Error> {
        let string = d.read_str()?;
        Ok(if !string.starts_with('#') {
            Ident::from_str(&string)
        } else { // FIXME(jseyfried): intercrate hygiene
            Ident::with_empty_ctxt(Symbol::gensym(&string[1..]))
        })
    }
}

/// A symbol is an interned or gensymed string. It's a NonZeroU32 so that
/// Option<Symbol> only takes up 4 bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(NonZeroU32);

// The interner is pointed to by a thread local value which is only set on the main thread
// with parallelization is disabled. So we don't allow `Symbol` to transfer between threads
// to avoid panics and other errors, even though it would be memory safe to do so.
#[cfg(not(parallel_queries))]
impl !Send for Symbol { }
#[cfg(not(parallel_queries))]
impl !Sync for Symbol { }

impl Symbol {
    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        with_interner(|interner| interner.intern(string))
    }

    pub fn interned(self) -> Self {
        with_interner(|interner| interner.interned(self))
    }

    /// Gensyms a new usize, using the current interner.
    pub fn gensym(string: &str) -> Self {
        with_interner(|interner| interner.gensym(string))
    }

    pub fn gensymed(self) -> Self {
        with_interner(|interner| interner.gensymed(self))
    }

    pub fn as_str(self) -> LocalInternedString {
        with_interner(|interner| unsafe {
            LocalInternedString {
                string: ::std::mem::transmute::<&str, &str>(interner.get(self))
            }
        })
    }

    pub fn as_interned_str(self) -> InternedString {
        with_interner(|interner| InternedString {
            symbol: interner.interned(self)
        })
    }

    #[inline(always)]
    pub fn as_u32(self) -> u32 {
        self.0.get()
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let is_gensymed = with_interner(|interner| interner.is_gensymed(*self));
        if is_gensymed {
            write!(f, "{}({})", self, self.0)
        } else {
            write!(f, "{}", self)
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.as_str(), f)
    }
}

impl Encodable for Symbol {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(&self.as_str())
    }
}

impl Decodable for Symbol {
    fn decode<D: Decoder>(d: &mut D) -> Result<Symbol, D::Error> {
        Ok(Symbol::intern(&d.read_str()?))
    }
}

impl<T: ::std::ops::Deref<Target=str>> PartialEq<T> for Symbol {
    fn eq(&self, other: &T) -> bool {
        self.as_str() == other.deref()
    }
}

/// Symbols (which are 1-indexed) index into this (which is 0-indexed
/// internally). The methods handle the index conversions.
#[derive(Default)]
pub struct SymbolVec(Vec<&'static str>);

impl SymbolVec {
    #[inline]
    fn new_symbol(&mut self, s: &'static str) -> Symbol {
        self.0.push(s);
        // self.0.len() cannot be zero because of the push above.
        Symbol(unsafe { NonZeroU32::new_unchecked(self.0.len() as u32) })
    }

    #[inline]
    fn get(&self, sym: Symbol) -> Option<&&'static str> {
        self.0.get(sym.0.get() as usize - 1)
    }

    #[inline]
    fn contains(&self, sym: Symbol) -> bool {
        sym.0.get() as usize <= self.0.len()
    }
}

// The `&'static str`s in this type actually point into the arena.
#[derive(Default)]
pub struct Interner {
    arena: DroplessArena,
    names: FxHashMap<&'static str, Symbol>,
    strings: SymbolVec,
    gensyms: Vec<Symbol>,
}

impl Interner {
    fn prefill(init: &[&str]) -> Self {
        let mut this = Interner::default();
        for &string in init {
            if string == "" {
                // We can't allocate empty strings in the arena, so handle this here.
                let name = this.strings.new_symbol("");
                this.names.insert("", name);
            } else {
                this.intern(string);
            }
        }
        this
    }

    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        // `from_utf8_unchecked` is safe since we just allocated a `&str` which is known to be
        // UTF-8.
        let string: &str = unsafe {
            str::from_utf8_unchecked(self.arena.alloc_slice(string.as_bytes()))
        };
        // It is safe to extend the arena allocation to `'static` because we only access
        // these while the arena is still alive.
        let string: &'static str =  unsafe {
            &*(string as *const str)
        };

        let name = self.strings.new_symbol(string);
        self.names.insert(string, name);
        name
    }

    pub fn interned(&self, symbol: Symbol) -> Symbol {
        if self.strings.contains(symbol) {
            symbol
        } else {
            self.interned(self.gensyms[(!0 - symbol.as_u32()) as usize])
        }
    }

    fn gensym(&mut self, string: &str) -> Symbol {
        let symbol = self.intern(string);
        self.gensymed(symbol)
    }

    fn gensymed(&mut self, symbol: Symbol) -> Symbol {
        self.gensyms.push(symbol);
        Symbol(NonZeroU32::new(!0 - self.gensyms.len() as u32 + 1).unwrap())
    }

    fn is_gensymed(&mut self, symbol: Symbol) -> bool {
        !self.strings.contains(symbol)
    }

    pub fn get(&self, symbol: Symbol) -> &str {
        match self.strings.get(symbol) {
            Some(string) => string,
            None => self.get(self.gensyms[(!0 - symbol.as_u32()) as usize]),
        }
    }
}

// In this macro, there is the requirement that the name (the number) must be monotonically
// increasing by one in the special identifiers, starting at 0; the same holds for the keywords,
// except starting from the next number instead of zero.
macro_rules! declare_keywords {(
    $( ($index: expr, $konst: ident, $string: expr) )*
) => {
    pub mod keywords {
        use super::{Symbol, Ident};
        use std::num::NonZeroU32;

        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct Keyword {
            ident: Ident,
        }
        impl Keyword {
            #[inline] pub fn ident(self) -> Ident { self.ident }
            #[inline] pub fn name(self) -> Symbol { self.ident.name }
        }
        // We must use `NonZeroU32::new_unchecked` below because it's `const`
        // and `NonZeroU32::new` is not. So we static_assert the non-zeroness
        // here.
        mod asserts {
            $(static_assert!($konst: $index > 0u32);)*
        }
        $(
            #[allow(non_upper_case_globals)]
            pub const $konst: Keyword = Keyword {
                ident: Ident::with_empty_ctxt(
                    super::Symbol(unsafe { NonZeroU32::new_unchecked($index) }))
            };
        )*

        impl ::std::str::FromStr for Keyword {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, ()> {
                match s {
                    $($string => Ok($konst),)*
                    _ => Err(()),
                }
            }
        }
    }

    impl Interner {
        pub fn fresh() -> Self {
            Interner::prefill(&[$($string,)*])
        }
    }
}}

// N.B., leaving holes in the ident table is bad! a different ident will get
// interned with the id from the hole, but it will be between the min and max
// of the reserved words, and thus tagged as "reserved".
// After modifying this list adjust `is_special`, `is_used_keyword`/`is_unused_keyword`,
// this should be rarely necessary though if the keywords are kept in alphabetic order.
declare_keywords! {
    // Special reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    // (0 cannot be used because Symbol uses NonZeroU32)
    (1,  Invalid,            "")
    (2,  PathRoot,           "{{root}}")
    (3,  DollarCrate,        "$crate")
    (4,  Underscore,         "_")

    // Keywords that are used in stable Rust.
    (5,  As,                 "as")
    (6,  Box,                "box")
    (7,  Break,              "break")
    (8,  Const,              "const")
    (9,  Continue,           "continue")
    (10,  Crate,              "crate")
    (11, Else,               "else")
    (12, Enum,               "enum")
    (13, Extern,             "extern")
    (14, False,              "false")
    (15, Fn,                 "fn")
    (16, For,                "for")
    (17, If,                 "if")
    (18, Impl,               "impl")
    (19, In,                 "in")
    (20, Let,                "let")
    (21, Loop,               "loop")
    (22, Match,              "match")
    (23, Mod,                "mod")
    (24, Move,               "move")
    (25, Mut,                "mut")
    (26, Pub,                "pub")
    (27, Ref,                "ref")
    (28, Return,             "return")
    (29, SelfLower,          "self")
    (30, SelfUpper,          "Self")
    (31, Static,             "static")
    (32, Struct,             "struct")
    (33, Super,              "super")
    (34, Trait,              "trait")
    (35, True,               "true")
    (36, Type,               "type")
    (37, Unsafe,             "unsafe")
    (38, Use,                "use")
    (39, Where,              "where")
    (40, While,              "while")

    // Keywords that are used in unstable Rust or reserved for future use.
    (41, Abstract,           "abstract")
    (42, Become,             "become")
    (43, Do,                 "do")
    (44, Final,              "final")
    (45, Macro,              "macro")
    (46, Override,           "override")
    (47, Priv,               "priv")
    (48, Typeof,             "typeof")
    (49, Unsized,            "unsized")
    (50, Virtual,            "virtual")
    (51, Yield,              "yield")

    // Edition-specific keywords that are used in stable Rust.
    (52, Dyn,                "dyn") // >= 2018 Edition only

    // Edition-specific keywords that are used in unstable Rust or reserved for future use.
    (53, Async,              "async") // >= 2018 Edition only
    (54, Try,                "try") // >= 2018 Edition only

    // Special lifetime names
    (55, UnderscoreLifetime, "'_")
    (56, StaticLifetime,     "'static")

    // Weak keywords, have special meaning only in specific contexts.
    (57, Auto,               "auto")
    (58, Catch,              "catch")
    (59, Default,            "default")
    (60, Existential,        "existential")
    (61, Union,              "union")
}

impl Symbol {
    fn is_used_keyword_2018(self) -> bool {
        self == keywords::Dyn.name()
    }

    fn is_unused_keyword_2018(self) -> bool {
        self >= keywords::Async.name() && self <= keywords::Try.name()
    }
}

impl Ident {
    // Returns `true` for reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    pub fn is_special(self) -> bool {
        self.name <= keywords::Underscore.name()
    }

    /// Returns `true` if the token is a keyword used in the language.
    pub fn is_used_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name >= keywords::As.name() && self.name <= keywords::While.name() ||
        self.name.is_used_keyword_2018() && self.span.rust_2018()
    }

    /// Returns `true` if the token is a keyword reserved for possible future use.
    pub fn is_unused_keyword(self) -> bool {
        // Note: `span.edition()` is relatively expensive, don't call it unless necessary.
        self.name >= keywords::Abstract.name() && self.name <= keywords::Yield.name() ||
        self.name.is_unused_keyword_2018() && self.span.rust_2018()
    }

    /// Returns `true` if the token is either a special identifier or a keyword.
    pub fn is_reserved(self) -> bool {
        self.is_special() || self.is_used_keyword() || self.is_unused_keyword()
    }

    /// A keyword or reserved identifier that can be used as a path segment.
    pub fn is_path_segment_keyword(self) -> bool {
        self.name == keywords::Super.name() ||
        self.name == keywords::SelfLower.name() ||
        self.name == keywords::SelfUpper.name() ||
        self.name == keywords::Extern.name() ||
        self.name == keywords::Crate.name() ||
        self.name == keywords::PathRoot.name() ||
        self.name == keywords::DollarCrate.name()
    }

    // We see this identifier in a normal identifier position, like variable name or a type.
    // How was it written originally? Did it use the raw form? Let's try to guess.
    pub fn is_raw_guess(self) -> bool {
        self.name != keywords::Invalid.name() && self.name != keywords::Underscore.name() &&
        self.is_reserved() && !self.is_path_segment_keyword()
    }
}

// If an interner exists, return it. Otherwise, prepare a fresh one.
#[inline]
fn with_interner<T, F: FnOnce(&mut Interner) -> T>(f: F) -> T {
    GLOBALS.with(|globals| f(&mut *globals.symbol_interner.lock()))
}

/// Represents a string stored in the interner. Because the interner outlives any thread
/// which uses this type, we can safely treat `string` which points to interner data,
/// as an immortal string, as long as this type never crosses between threads.
// FIXME: ensure that the interner outlives any thread which uses `LocalInternedString`,
// by creating a new thread right after constructing the interner.
#[derive(Clone, Copy, Hash, PartialOrd, Eq, Ord)]
pub struct LocalInternedString {
    string: &'static str,
}

impl LocalInternedString {
    pub fn as_interned_str(self) -> InternedString {
        InternedString {
            symbol: Symbol::intern(self.string)
        }
    }

    pub fn get(&self) -> &'static str {
        self.string
    }
}

impl<U: ?Sized> ::std::convert::AsRef<U> for LocalInternedString
where
    str: ::std::convert::AsRef<U>
{
    fn as_ref(&self) -> &U {
        self.string.as_ref()
    }
}

impl<T: ::std::ops::Deref<Target = str>> ::std::cmp::PartialEq<T> for LocalInternedString {
    fn eq(&self, other: &T) -> bool {
        self.string == other.deref()
    }
}

impl ::std::cmp::PartialEq<LocalInternedString> for str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<LocalInternedString> for &'a str {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

impl ::std::cmp::PartialEq<LocalInternedString> for String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<LocalInternedString> for &'a String {
    fn eq(&self, other: &LocalInternedString) -> bool {
        *self == other.string
    }
}

impl !Send for LocalInternedString {}
impl !Sync for LocalInternedString {}

impl ::std::ops::Deref for LocalInternedString {
    type Target = str;
    fn deref(&self) -> &str { self.string }
}

impl fmt::Debug for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.string, f)
    }
}

impl fmt::Display for LocalInternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.string, f)
    }
}

impl Decodable for LocalInternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<LocalInternedString, D::Error> {
        Ok(Symbol::intern(&d.read_str()?).as_str())
    }
}

impl Encodable for LocalInternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self.string)
    }
}

/// Represents a string stored in the string interner.
#[derive(Clone, Copy, Eq)]
pub struct InternedString {
    symbol: Symbol,
}

impl InternedString {
    pub fn with<F: FnOnce(&str) -> R, R>(self, f: F) -> R {
        let str = with_interner(|interner| {
            interner.get(self.symbol) as *const str
        });
        // This is safe because the interner keeps string alive until it is dropped.
        // We can access it because we know the interner is still alive since we use a
        // scoped thread local to access it, and it was alive at the beginning of this scope
        unsafe { f(&*str) }
    }

    pub fn as_symbol(self) -> Symbol {
        self.symbol
    }

    pub fn as_str(self) -> LocalInternedString {
        self.symbol.as_str()
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.with(|str| str.hash(state))
    }
}

impl PartialOrd<InternedString> for InternedString {
    fn partial_cmp(&self, other: &InternedString) -> Option<Ordering> {
        if self.symbol == other.symbol {
            return Some(Ordering::Equal);
        }
        self.with(|self_str| other.with(|other_str| self_str.partial_cmp(other_str)))
    }
}

impl Ord for InternedString {
    fn cmp(&self, other: &InternedString) -> Ordering {
        if self.symbol == other.symbol {
            return Ordering::Equal;
        }
        self.with(|self_str| other.with(|other_str| self_str.cmp(&other_str)))
    }
}

impl<T: ::std::ops::Deref<Target = str>> PartialEq<T> for InternedString {
    fn eq(&self, other: &T) -> bool {
        self.with(|string| string == other.deref())
    }
}

impl PartialEq<InternedString> for InternedString {
    fn eq(&self, other: &InternedString) -> bool {
        self.symbol == other.symbol
    }
}

impl PartialEq<InternedString> for str {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| self == string)
    }
}

impl<'a> PartialEq<InternedString> for &'a str {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| *self == string)
    }
}

impl PartialEq<InternedString> for String {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| self == string)
    }
}

impl<'a> PartialEq<InternedString> for &'a String {
    fn eq(&self, other: &InternedString) -> bool {
        other.with(|string| *self == string)
    }
}

impl ::std::convert::From<InternedString> for String {
    fn from(val: InternedString) -> String {
        val.as_symbol().to_string()
    }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.with(|str| fmt::Debug::fmt(&str, f))
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.with(|str| fmt::Display::fmt(&str, f))
    }
}

impl Decodable for InternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<InternedString, D::Error> {
        Ok(Symbol::intern(&d.read_str()?).as_interned_str())
    }
}

impl Encodable for InternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        self.with(|string| s.emit_str(string))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Globals;

    #[test]
    fn interner_tests() {
        let mut i: Interner = Interner::default();
        let nz = |n| NonZeroU32::new(n).unwrap();

        // first one is 1:
        assert_eq!(i.intern("dog"), Symbol(nz(1)));
        // re-use gets the same entry:
        assert_eq!(i.intern("dog"), Symbol(nz(1)));
        // different string gets a different number:
        assert_eq!(i.intern("cat"), Symbol(nz(2)));
        assert_eq!(i.intern("cat"), Symbol(nz(2)));
        // dog is still at 1
        assert_eq!(i.intern("dog"), Symbol(nz(1)));
        assert_eq!(i.gensym("zebra"), Symbol(nz(4294967295)));
        // gensym of same string gets new number:
        assert_eq!(i.gensym("zebra"), Symbol(nz(4294967294)));
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), Symbol(nz(4294967293)));
    }

    #[test]
    fn without_first_quote_test() {
        GLOBALS.set(&Globals::new(), || {
            let i = Ident::from_str("'break");
            assert_eq!(i.without_first_quote().name, keywords::Break.name());
        });
    }
}
