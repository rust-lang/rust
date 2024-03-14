//! A simplified version of quote-crate like quasi quote macro
#![allow(clippy::crate_in_macro_def)]

use span::Span;
use syntax::format_smolstr;

use crate::name::Name;

pub(crate) const fn dollar_crate(span: Span) -> tt::Ident<Span> {
    tt::Ident { text: syntax::SmolStr::new_static("$crate"), span }
}

// A helper macro quote macro
// FIXME:
// 1. Not all puncts are handled
// 2. #()* pattern repetition not supported now
//    * But we can do it manually, see `test_quote_derive_copy_hack`
#[doc(hidden)]
#[macro_export]
macro_rules! __quote {
    ($span:ident) => {
        Vec::<$crate::tt::TokenTree>::new()
    };

    ( @SUBTREE($span:ident) $delim:ident $($tt:tt)* ) => {
        {
            let children = $crate::__quote!($span $($tt)*);
            $crate::tt::Subtree {
                delimiter: crate::tt::Delimiter {
                    kind: crate::tt::DelimiterKind::$delim,
                    open: $span,
                    close: $span,
                },
                token_trees: $crate::quote::IntoTt::to_tokens(children).into_boxed_slice(),
            }
        }
    };

    ( @PUNCT($span:ident) $first:literal ) => {
        {
            vec![
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $first,
                    spacing: crate::tt::Spacing::Alone,
                    span: $span,
                }).into()
            ]
        }
    };

    ( @PUNCT($span:ident) $first:literal, $sec:literal ) => {
        {
            vec![
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $first,
                    spacing: crate::tt::Spacing::Joint,
                    span: $span,
                }).into(),
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $sec,
                    spacing: crate::tt::Spacing::Alone,
                    span: $span,
                }).into()
            ]
        }
    };

    // hash variable
    ($span:ident # $first:ident $($tail:tt)* ) => {
        {
            let token = $crate::quote::ToTokenTree::to_token($first, $span);
            let mut tokens = vec![token.into()];
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($span $($tail)*));
            tokens.append(&mut tail_tokens);
            tokens
        }
    };

    ($span:ident ## $first:ident $($tail:tt)* ) => {
        {
            let mut tokens = $first.into_iter().map(|it| $crate::quote::ToTokenTree::to_token(it, $span)).collect::<Vec<crate::tt::TokenTree>>();
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($span $($tail)*));
            tokens.append(&mut tail_tokens);
            tokens
        }
    };

    // Brace
    ($span:ident  { $($tt:tt)* } ) => { $crate::__quote!(@SUBTREE($span) Brace $($tt)*) };
    // Bracket
    ($span:ident  [ $($tt:tt)* ] ) => { $crate::__quote!(@SUBTREE($span) Bracket $($tt)*) };
    // Parenthesis
    ($span:ident  ( $($tt:tt)* ) ) => { $crate::__quote!(@SUBTREE($span) Parenthesis $($tt)*) };

    // Literal
    ($span:ident $tt:literal ) => { vec![$crate::quote::ToTokenTree::to_token($tt, $span).into()] };
    // Ident
    ($span:ident $tt:ident ) => {
        vec![ {
            crate::tt::Leaf::Ident(crate::tt::Ident {
                text: stringify!($tt).into(),
                span: $span,
            }).into()
        }]
    };

    // Puncts
    // FIXME: Not all puncts are handled
    ($span:ident -> ) => {$crate::__quote!(@PUNCT($span) '-', '>')};
    ($span:ident & ) => {$crate::__quote!(@PUNCT($span) '&')};
    ($span:ident , ) => {$crate::__quote!(@PUNCT($span) ',')};
    ($span:ident : ) => {$crate::__quote!(@PUNCT($span) ':')};
    ($span:ident ; ) => {$crate::__quote!(@PUNCT($span) ';')};
    ($span:ident :: ) => {$crate::__quote!(@PUNCT($span) ':', ':')};
    ($span:ident . ) => {$crate::__quote!(@PUNCT($span) '.')};
    ($span:ident < ) => {$crate::__quote!(@PUNCT($span) '<')};
    ($span:ident > ) => {$crate::__quote!(@PUNCT($span) '>')};
    ($span:ident ! ) => {$crate::__quote!(@PUNCT($span) '!')};

    ($span:ident $first:tt $($tail:tt)+ ) => {
        {
            let mut tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($span $first ));
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($span $($tail)*));

            tokens.append(&mut tail_tokens);
            tokens
        }
    };
}

/// FIXME:
/// It probably should implement in proc-macro
#[macro_export]
macro_rules! quote {
    ($span:ident=> $($tt:tt)* ) => {
        $crate::quote::IntoTt::to_subtree($crate::__quote!($span $($tt)*), $span)
    }
}

pub(crate) trait IntoTt {
    fn to_subtree(self, span: Span) -> crate::tt::Subtree;
    fn to_tokens(self) -> Vec<crate::tt::TokenTree>;
}

impl IntoTt for Vec<crate::tt::TokenTree> {
    fn to_subtree(self, span: Span) -> crate::tt::Subtree {
        crate::tt::Subtree {
            delimiter: crate::tt::Delimiter::invisible_spanned(span),
            token_trees: self.into_boxed_slice(),
        }
    }

    fn to_tokens(self) -> Vec<crate::tt::TokenTree> {
        self
    }
}

impl IntoTt for crate::tt::Subtree {
    fn to_subtree(self, _: Span) -> crate::tt::Subtree {
        self
    }

    fn to_tokens(self) -> Vec<crate::tt::TokenTree> {
        vec![crate::tt::TokenTree::Subtree(self)]
    }
}

pub(crate) trait ToTokenTree {
    fn to_token(self, span: Span) -> crate::tt::TokenTree;
}

impl ToTokenTree for crate::tt::TokenTree {
    fn to_token(self, _: Span) -> crate::tt::TokenTree {
        self
    }
}

impl ToTokenTree for &crate::tt::TokenTree {
    fn to_token(self, _: Span) -> crate::tt::TokenTree {
        self.clone()
    }
}

impl ToTokenTree for crate::tt::Subtree {
    fn to_token(self, _: Span) -> crate::tt::TokenTree {
        self.into()
    }
}

macro_rules! impl_to_to_tokentrees {
    ($($span:ident: $ty:ty => $this:ident $im:block;)*) => {
        $(
            impl ToTokenTree for $ty {
                fn to_token($this, $span: Span) -> crate::tt::TokenTree {
                    let leaf: crate::tt::Leaf = $im.into();
                    leaf.into()
                }
            }

            impl ToTokenTree for &$ty {
                fn to_token($this, $span: Span) -> crate::tt::TokenTree {
                    let leaf: crate::tt::Leaf = $im.clone().into();
                    leaf.into()
                }
            }
        )*
    }
}

impl_to_to_tokentrees! {
    span: u32 => self { crate::tt::Literal{text: self.to_string().into(), span} };
    span: usize => self { crate::tt::Literal{text: self.to_string().into(), span} };
    span: i32 => self { crate::tt::Literal{text: self.to_string().into(), span} };
    span: bool => self { crate::tt::Ident{text: self.to_string().into(), span} };
    _span: crate::tt::Leaf => self { self };
    _span: crate::tt::Literal => self { self };
    _span: crate::tt::Ident => self { self };
    _span: crate::tt::Punct => self { self };
    span: &str => self { crate::tt::Literal{text: format_smolstr!("\"{}\"", self.escape_default()), span}};
    span: String => self { crate::tt::Literal{text: format_smolstr!("\"{}\"", self.escape_default()), span}};
    span: Name => self { crate::tt::Ident{text: self.to_smol_str(), span}};
}

#[cfg(test)]
mod tests {
    use crate::tt;
    use base_db::FileId;
    use expect_test::expect;
    use span::{SpanAnchor, SyntaxContextId, ROOT_ERASED_FILE_AST_ID};
    use syntax::{TextRange, TextSize};

    const DUMMY: tt::Span = tt::Span {
        range: TextRange::empty(TextSize::new(0)),
        anchor: SpanAnchor { file_id: FileId::BOGUS, ast_id: ROOT_ERASED_FILE_AST_ID },
        ctx: SyntaxContextId::ROOT,
    };

    #[test]
    fn test_quote_delimiters() {
        assert_eq!(quote!(DUMMY =>{}).to_string(), "{}");
        assert_eq!(quote!(DUMMY =>()).to_string(), "()");
        assert_eq!(quote!(DUMMY =>[]).to_string(), "[]");
    }

    #[test]
    fn test_quote_idents() {
        assert_eq!(quote!(DUMMY =>32).to_string(), "32");
        assert_eq!(quote!(DUMMY =>struct).to_string(), "struct");
    }

    #[test]
    fn test_quote_hash_simple_literal() {
        let a = 20;
        assert_eq!(quote!(DUMMY =>#a).to_string(), "20");
        let s: String = "hello".into();
        assert_eq!(quote!(DUMMY =>#s).to_string(), "\"hello\"");
    }

    fn mk_ident(name: &str) -> crate::tt::Ident {
        crate::tt::Ident { text: name.into(), span: DUMMY }
    }

    #[test]
    fn test_quote_hash_token_tree() {
        let a = mk_ident("hello");

        let quoted = quote!(DUMMY =>#a);
        assert_eq!(quoted.to_string(), "hello");
        let t = format!("{quoted:#?}");
        expect![[r#"
            SUBTREE $$ 937550:0@0..0#0 937550:0@0..0#0
              IDENT   hello 937550:0@0..0#0"#]]
        .assert_eq(&t);
    }

    #[test]
    fn test_quote_simple_derive_copy() {
        let name = mk_ident("Foo");

        let quoted = quote! {DUMMY =>
            impl Clone for #name {
                fn clone(&self) -> Self {
                    Self {}
                }
            }
        };

        assert_eq!(quoted.to_string(), "impl Clone for Foo {fn clone (& self) -> Self {Self {}}}");
    }

    #[test]
    fn test_quote_derive_copy_hack() {
        // Assume the given struct is:
        // struct Foo {
        //  name: String,
        //  id: u32,
        // }
        let struct_name = mk_ident("Foo");
        let fields = [mk_ident("name"), mk_ident("id")];
        let fields = fields
            .iter()
            .flat_map(|it| quote!(DUMMY =>#it: self.#it.clone(), ).token_trees.into_vec());

        let list = crate::tt::Subtree {
            delimiter: crate::tt::Delimiter {
                kind: crate::tt::DelimiterKind::Brace,
                open: DUMMY,
                close: DUMMY,
            },
            token_trees: fields.collect(),
        };

        let quoted = quote! {DUMMY =>
            impl Clone for #struct_name {
                fn clone(&self) -> Self {
                    Self #list
                }
            }
        };

        assert_eq!(quoted.to_string(), "impl Clone for Foo {fn clone (& self) -> Self {Self {name : self . name . clone () , id : self . id . clone () ,}}}");
    }
}
