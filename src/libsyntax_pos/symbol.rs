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
//! allows bidirectional lookup; i.e. given a value, one can easily find the
//! type, and vice versa.

use hygiene::SyntaxContext;

use serialize::{Decodable, Decoder, Encodable, Encoder};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Ident {
    pub name: Symbol,
    pub ctxt: SyntaxContext,
}

impl Ident {
    pub const fn with_empty_ctxt(name: Symbol) -> Ident {
        Ident { name: name, ctxt: SyntaxContext::empty() }
    }

    /// Maps a string to an identifier with an empty syntax context.
    pub fn from_str(string: &str) -> Ident {
        Ident::with_empty_ctxt(Symbol::intern(string))
    }

    pub fn modern(self) -> Ident {
        Ident { name: self.name, ctxt: self.ctxt.modern() }
    }
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{:?}", self.name, self.ctxt)
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl Encodable for Ident {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        if self.ctxt.modern() == SyntaxContext::empty() {
            s.emit_str(&self.name.as_str())
        } else { // FIXME(jseyfried) intercrate hygiene
            let mut string = "#".to_owned();
            string.push_str(&self.name.as_str());
            s.emit_str(&string)
        }
    }
}

impl Decodable for Ident {
    fn decode<D: Decoder>(d: &mut D) -> Result<Ident, D::Error> {
        let string = d.read_str()?;
        Ok(if !string.starts_with('#') {
            Ident::from_str(&string)
        } else { // FIXME(jseyfried) intercrate hygiene
            Ident::with_empty_ctxt(Symbol::gensym(&string[1..]))
        })
    }
}

/// A symbol is an interned or gensymed string.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Symbol(u32);

// The interner in thread-local, so `Symbol` shouldn't move between threads.
impl !Send for Symbol { }
impl !Sync for Symbol { }

impl Symbol {
    /// Maps a string to its interned representation.
    pub fn intern(string: &str) -> Self {
        with_interner(|interner| interner.intern(string))
    }

    pub fn interned(self) -> Self {
        with_interner(|interner| interner.interned(self))
    }

    /// gensym's a new usize, using the current interner.
    pub fn gensym(string: &str) -> Self {
        with_interner(|interner| interner.gensym(string))
    }

    pub fn gensymed(self) -> Self {
        with_interner(|interner| interner.gensymed(self))
    }

    pub fn as_str(self) -> InternedString {
        with_interner(|interner| unsafe {
            InternedString {
                string: ::std::mem::transmute::<&str, &str>(interner.get(self))
            }
        })
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl<'a> From<&'a str> for Symbol {
    fn from(string: &'a str) -> Symbol {
        Symbol::intern(string)
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({})", self, self.0)
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

#[derive(Default)]
pub struct Interner {
    names: HashMap<Box<str>, Symbol>,
    strings: Vec<Box<str>>,
    gensyms: Vec<Symbol>,
}

impl Interner {
    pub fn new() -> Self {
        Interner::default()
    }

    fn prefill(init: &[&str]) -> Self {
        let mut this = Interner::new();
        for &string in init {
            this.intern(string);
        }
        this
    }

    pub fn intern(&mut self, string: &str) -> Symbol {
        if let Some(&name) = self.names.get(string) {
            return name;
        }

        let name = Symbol(self.strings.len() as u32);
        let string = string.to_string().into_boxed_str();
        self.strings.push(string.clone());
        self.names.insert(string, name);
        name
    }

    pub fn interned(&self, symbol: Symbol) -> Symbol {
        if (symbol.0 as usize) < self.strings.len() {
            symbol
        } else {
            self.interned(self.gensyms[(!0 - symbol.0) as usize])
        }
    }

    fn gensym(&mut self, string: &str) -> Symbol {
        let symbol = self.intern(string);
        self.gensymed(symbol)
    }

    fn gensymed(&mut self, symbol: Symbol) -> Symbol {
        self.gensyms.push(symbol);
        Symbol(!0 - self.gensyms.len() as u32 + 1)
    }

    pub fn get(&self, symbol: Symbol) -> &str {
        match self.strings.get(symbol.0 as usize) {
            Some(ref string) => string,
            None => self.get(self.gensyms[(!0 - symbol.0) as usize]),
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
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub struct Keyword {
            ident: Ident,
        }
        impl Keyword {
            #[inline] pub fn ident(self) -> Ident { self.ident }
            #[inline] pub fn name(self) -> Symbol { self.ident.name }
        }
        $(
            #[allow(non_upper_case_globals)]
            pub const $konst: Keyword = Keyword {
                ident: Ident::with_empty_ctxt(super::Symbol($index))
            };
        )*
    }

    impl Interner {
        fn fresh() -> Self {
            Interner::prefill(&[$($string,)*])
        }
    }
}}

// NB: leaving holes in the ident table is bad! a different ident will get
// interned with the id from the hole, but it will be between the min and max
// of the reserved words, and thus tagged as "reserved".
// After modifying this list adjust `is_special_ident`, `is_used_keyword`/`is_unused_keyword`,
// this should be rarely necessary though if the keywords are kept in alphabetic order.
declare_keywords! {
    // Special reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    (0,  Invalid,        "")
    (1,  CrateRoot,      "{{root}}")
    (2,  DollarCrate,    "$crate")

    // Keywords used in the language.
    (3,  As,             "as")
    (4,  Box,            "box")
    (5,  Break,          "break")
    (6,  Const,          "const")
    (7,  Continue,       "continue")
    (8,  Crate,          "crate")
    (9,  Else,           "else")
    (10, Enum,           "enum")
    (11, Extern,         "extern")
    (12, False,          "false")
    (13, Fn,             "fn")
    (14, For,            "for")
    (15, If,             "if")
    (16, Impl,           "impl")
    (17, In,             "in")
    (18, Let,            "let")
    (19, Loop,           "loop")
    (20, Match,          "match")
    (21, Mod,            "mod")
    (22, Move,           "move")
    (23, Mut,            "mut")
    (24, Pub,            "pub")
    (25, Ref,            "ref")
    (26, Return,         "return")
    (27, SelfValue,      "self")
    (28, SelfType,       "Self")
    (29, Static,         "static")
    (30, Struct,         "struct")
    (31, Super,          "super")
    (32, Trait,          "trait")
    (33, True,           "true")
    (34, Type,           "type")
    (35, Unsafe,         "unsafe")
    (36, Use,            "use")
    (37, Where,          "where")
    (38, While,          "while")

    // Keywords reserved for future use.
    (39, Abstract,       "abstract")
    (40, Alignof,        "alignof")
    (41, Become,         "become")
    (42, Do,             "do")
    (43, Final,          "final")
    (44, Macro,          "macro")
    (45, Offsetof,       "offsetof")
    (46, Override,       "override")
    (47, Priv,           "priv")
    (48, Proc,           "proc")
    (49, Pure,           "pure")
    (50, Sizeof,         "sizeof")
    (51, Typeof,         "typeof")
    (52, Unsized,        "unsized")
    (53, Virtual,        "virtual")
    (54, Yield,          "yield")

    // Weak keywords, have special meaning only in specific contexts.
    (55, Auto,           "auto")
    (56, Catch,          "catch")
    (57, Default,        "default")
    (58, Dyn,            "dyn")
    (59, StaticLifetime, "'static")
    (60, Union,          "union")
}

// If an interner exists in TLS, return it. Otherwise, prepare a fresh one.
fn with_interner<T, F: FnOnce(&mut Interner) -> T>(f: F) -> T {
    thread_local!(static INTERNER: RefCell<Interner> = {
        RefCell::new(Interner::fresh())
    });
    INTERNER.with(|interner| f(&mut *interner.borrow_mut()))
}

/// Represents a string stored in the thread-local interner. Because the
/// interner lives for the life of the thread, this can be safely treated as an
/// immortal string, as long as it never crosses between threads.
///
/// FIXME(pcwalton): You must be careful about what you do in the destructors
/// of objects stored in TLS, because they may run after the interner is
/// destroyed. In particular, they must not access string contents. This can
/// be fixed in the future by just leaking all strings until thread death
/// somehow.
#[derive(Clone, Copy, Hash, PartialOrd, Eq, Ord)]
pub struct InternedString {
    string: &'static str,
}

impl<U: ?Sized> ::std::convert::AsRef<U> for InternedString where str: ::std::convert::AsRef<U> {
    fn as_ref(&self) -> &U {
        self.string.as_ref()
    }
}

impl<T: ::std::ops::Deref<Target = str>> ::std::cmp::PartialEq<T> for InternedString {
    fn eq(&self, other: &T) -> bool {
        self.string == other.deref()
    }
}

impl ::std::cmp::PartialEq<InternedString> for str {
    fn eq(&self, other: &InternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<InternedString> for &'a str {
    fn eq(&self, other: &InternedString) -> bool {
        *self == other.string
    }
}

impl ::std::cmp::PartialEq<InternedString> for String {
    fn eq(&self, other: &InternedString) -> bool {
        self == other.string
    }
}

impl<'a> ::std::cmp::PartialEq<InternedString> for &'a String {
    fn eq(&self, other: &InternedString) -> bool {
        *self == other.string
    }
}

impl !Send for InternedString { }

impl ::std::ops::Deref for InternedString {
    type Target = str;
    fn deref(&self) -> &str { self.string }
}

impl fmt::Debug for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self.string, f)
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.string, f)
    }
}

impl Decodable for InternedString {
    fn decode<D: Decoder>(d: &mut D) -> Result<InternedString, D::Error> {
        Ok(Symbol::intern(&d.read_str()?).as_str())
    }
}

impl Encodable for InternedString {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_str(self.string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interner_tests() {
        let mut i: Interner = Interner::new();
        // first one is zero:
        assert_eq!(i.intern("dog"), Symbol(0));
        // re-use gets the same entry:
        assert_eq!(i.intern ("dog"), Symbol(0));
        // different string gets a different #:
        assert_eq!(i.intern("cat"), Symbol(1));
        assert_eq!(i.intern("cat"), Symbol(1));
        // dog is still at zero
        assert_eq!(i.intern("dog"), Symbol(0));
        assert_eq!(i.gensym("zebra"), Symbol(4294967295));
        // gensym of same string gets new number :
        assert_eq!(i.gensym("zebra"), Symbol(4294967294));
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), Symbol(4294967293));
    }
}
