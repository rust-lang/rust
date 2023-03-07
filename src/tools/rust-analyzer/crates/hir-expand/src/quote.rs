//! A simplified version of quote-crate like quasi quote macro

// A helper macro quote macro
// FIXME:
// 1. Not all puncts are handled
// 2. #()* pattern repetition not supported now
//    * But we can do it manually, see `test_quote_derive_copy_hack`
#[doc(hidden)]
#[macro_export]
macro_rules! __quote {
    () => {
        Vec::<crate::tt::TokenTree>::new()
    };

    ( @SUBTREE $delim:ident $($tt:tt)* ) => {
        {
            let children = $crate::__quote!($($tt)*);
            crate::tt::Subtree {
                delimiter: crate::tt::Delimiter {
                    kind: crate::tt::DelimiterKind::$delim,
                    open: crate::tt::TokenId::unspecified(),
                    close: crate::tt::TokenId::unspecified(),
                },
                token_trees: $crate::quote::IntoTt::to_tokens(children),
            }
        }
    };

    ( @PUNCT $first:literal ) => {
        {
            vec![
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $first,
                    spacing: crate::tt::Spacing::Alone,
                    span: crate::tt::TokenId::unspecified(),
                }).into()
            ]
        }
    };

    ( @PUNCT $first:literal, $sec:literal ) => {
        {
            vec![
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $first,
                    spacing: crate::tt::Spacing::Joint,
                    span: crate::tt::TokenId::unspecified(),
                }).into(),
                crate::tt::Leaf::Punct(crate::tt::Punct {
                    char: $sec,
                    spacing: crate::tt::Spacing::Alone,
                    span: crate::tt::TokenId::unspecified(),
                }).into()
            ]
        }
    };

    // hash variable
    ( # $first:ident $($tail:tt)* ) => {
        {
            let token = $crate::quote::ToTokenTree::to_token($first);
            let mut tokens = vec![token.into()];
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($($tail)*));
            tokens.append(&mut tail_tokens);
            tokens
        }
    };

    ( ## $first:ident $($tail:tt)* ) => {
        {
            let mut tokens = $first.into_iter().map($crate::quote::ToTokenTree::to_token).collect::<Vec<crate::tt::TokenTree>>();
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($($tail)*));
            tokens.append(&mut tail_tokens);
            tokens
        }
    };

    // Brace
    ( { $($tt:tt)* } ) => { $crate::__quote!(@SUBTREE Brace $($tt)*) };
    // Bracket
    ( [ $($tt:tt)* ] ) => { $crate::__quote!(@SUBTREE Bracket $($tt)*) };
    // Parenthesis
    ( ( $($tt:tt)* ) ) => { $crate::__quote!(@SUBTREE Parenthesis $($tt)*) };

    // Literal
    ( $tt:literal ) => { vec![$crate::quote::ToTokenTree::to_token($tt).into()] };
    // Ident
    ( $tt:ident ) => {
        vec![ {
            crate::tt::Leaf::Ident(crate::tt::Ident {
                text: stringify!($tt).into(),
                span: crate::tt::TokenId::unspecified(),
            }).into()
        }]
    };

    // Puncts
    // FIXME: Not all puncts are handled
    ( -> ) => {$crate::__quote!(@PUNCT '-', '>')};
    ( & ) => {$crate::__quote!(@PUNCT '&')};
    ( , ) => {$crate::__quote!(@PUNCT ',')};
    ( : ) => {$crate::__quote!(@PUNCT ':')};
    ( ; ) => {$crate::__quote!(@PUNCT ';')};
    ( :: ) => {$crate::__quote!(@PUNCT ':', ':')};
    ( . ) => {$crate::__quote!(@PUNCT '.')};
    ( < ) => {$crate::__quote!(@PUNCT '<')};
    ( > ) => {$crate::__quote!(@PUNCT '>')};
    ( ! ) => {$crate::__quote!(@PUNCT '!')};

    ( $first:tt $($tail:tt)+ ) => {
        {
            let mut tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($first));
            let mut tail_tokens = $crate::quote::IntoTt::to_tokens($crate::__quote!($($tail)*));

            tokens.append(&mut tail_tokens);
            tokens
        }
    };
}

/// FIXME:
/// It probably should implement in proc-macro
#[macro_export]
macro_rules! quote {
    ( $($tt:tt)* ) => {
        $crate::quote::IntoTt::to_subtree($crate::__quote!($($tt)*))
    }
}

pub(crate) trait IntoTt {
    fn to_subtree(self) -> crate::tt::Subtree;
    fn to_tokens(self) -> Vec<crate::tt::TokenTree>;
}

impl IntoTt for Vec<crate::tt::TokenTree> {
    fn to_subtree(self) -> crate::tt::Subtree {
        crate::tt::Subtree { delimiter: crate::tt::Delimiter::unspecified(), token_trees: self }
    }

    fn to_tokens(self) -> Vec<crate::tt::TokenTree> {
        self
    }
}

impl IntoTt for crate::tt::Subtree {
    fn to_subtree(self) -> crate::tt::Subtree {
        self
    }

    fn to_tokens(self) -> Vec<crate::tt::TokenTree> {
        vec![crate::tt::TokenTree::Subtree(self)]
    }
}

pub(crate) trait ToTokenTree {
    fn to_token(self) -> crate::tt::TokenTree;
}

impl ToTokenTree for crate::tt::TokenTree {
    fn to_token(self) -> crate::tt::TokenTree {
        self
    }
}

impl ToTokenTree for crate::tt::Subtree {
    fn to_token(self) -> crate::tt::TokenTree {
        self.into()
    }
}

macro_rules! impl_to_to_tokentrees {
    ($($ty:ty => $this:ident $im:block);*) => {
        $(
            impl ToTokenTree for $ty {
                fn to_token($this) -> crate::tt::TokenTree {
                    let leaf: crate::tt::Leaf = $im.into();
                    leaf.into()
                }
            }

            impl ToTokenTree for &$ty {
                fn to_token($this) -> crate::tt::TokenTree {
                    let leaf: crate::tt::Leaf = $im.clone().into();
                    leaf.into()
                }
            }
        )*
    }
}

impl_to_to_tokentrees! {
    u32 => self { crate::tt::Literal{text: self.to_string().into(), span: crate::tt::TokenId::unspecified()} };
    usize => self { crate::tt::Literal{text: self.to_string().into(), span: crate::tt::TokenId::unspecified()} };
    i32 => self { crate::tt::Literal{text: self.to_string().into(), span: crate::tt::TokenId::unspecified()} };
    bool => self { crate::tt::Ident{text: self.to_string().into(), span: crate::tt::TokenId::unspecified()} };
    crate::tt::Leaf => self { self };
    crate::tt::Literal => self { self };
    crate::tt::Ident => self { self };
    crate::tt::Punct => self { self };
    &str => self { crate::tt::Literal{text: format!("\"{}\"", self.escape_default()).into(), span: crate::tt::TokenId::unspecified()}};
    String => self { crate::tt::Literal{text: format!("\"{}\"", self.escape_default()).into(), span: crate::tt::TokenId::unspecified()}}
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_quote_delimiters() {
        assert_eq!(quote!({}).to_string(), "{}");
        assert_eq!(quote!(()).to_string(), "()");
        assert_eq!(quote!([]).to_string(), "[]");
    }

    #[test]
    fn test_quote_idents() {
        assert_eq!(quote!(32).to_string(), "32");
        assert_eq!(quote!(struct).to_string(), "struct");
    }

    #[test]
    fn test_quote_hash_simple_literal() {
        let a = 20;
        assert_eq!(quote!(#a).to_string(), "20");
        let s: String = "hello".into();
        assert_eq!(quote!(#s).to_string(), "\"hello\"");
    }

    fn mk_ident(name: &str) -> crate::tt::Ident {
        crate::tt::Ident { text: name.into(), span: crate::tt::TokenId::unspecified() }
    }

    #[test]
    fn test_quote_hash_token_tree() {
        let a = mk_ident("hello");

        let quoted = quote!(#a);
        assert_eq!(quoted.to_string(), "hello");
        let t = format!("{quoted:?}");
        assert_eq!(t, "SUBTREE $$ 4294967295 4294967295\n  IDENT   hello 4294967295");
    }

    #[test]
    fn test_quote_simple_derive_copy() {
        let name = mk_ident("Foo");

        let quoted = quote! {
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
        let fields = fields.iter().flat_map(|it| quote!(#it: self.#it.clone(), ).token_trees);

        let list = crate::tt::Subtree {
            delimiter: crate::tt::Delimiter {
                kind: crate::tt::DelimiterKind::Brace,
                open: crate::tt::TokenId::unspecified(),
                close: crate::tt::TokenId::unspecified(),
            },
            token_trees: fields.collect(),
        };

        let quoted = quote! {
            impl Clone for #struct_name {
                fn clone(&self) -> Self {
                    Self #list
                }
            }
        };

        assert_eq!(quoted.to_string(), "impl Clone for Foo {fn clone (& self) -> Self {Self {name : self . name . clone () , id : self . id . clone () ,}}}");
    }
}
