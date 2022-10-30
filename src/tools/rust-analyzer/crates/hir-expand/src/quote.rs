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
        Vec::<tt::TokenTree>::new()
    };

    ( @SUBTREE $delim:ident $($tt:tt)* ) => {
        {
            let children = $crate::__quote!($($tt)*);
            tt::Subtree {
                delimiter: Some(tt::Delimiter {
                    kind: tt::DelimiterKind::$delim,
                    id: tt::TokenId::unspecified(),
                }),
                token_trees: $crate::quote::IntoTt::to_tokens(children),
            }
        }
    };

    ( @PUNCT $first:literal ) => {
        {
            vec![
                tt::Leaf::Punct(tt::Punct {
                    char: $first,
                    spacing: tt::Spacing::Alone,
                    id: tt::TokenId::unspecified(),
                }).into()
            ]
        }
    };

    ( @PUNCT $first:literal, $sec:literal ) => {
        {
            vec![
                tt::Leaf::Punct(tt::Punct {
                    char: $first,
                    spacing: tt::Spacing::Joint,
                    id: tt::TokenId::unspecified(),
                }).into(),
                tt::Leaf::Punct(tt::Punct {
                    char: $sec,
                    spacing: tt::Spacing::Alone,
                    id: tt::TokenId::unspecified(),
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
            let mut tokens = $first.into_iter().map($crate::quote::ToTokenTree::to_token).collect::<Vec<tt::TokenTree>>();
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
            tt::Leaf::Ident(tt::Ident {
                text: stringify!($tt).into(),
                id: tt::TokenId::unspecified(),
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
    fn to_subtree(self) -> tt::Subtree;
    fn to_tokens(self) -> Vec<tt::TokenTree>;
}

impl IntoTt for Vec<tt::TokenTree> {
    fn to_subtree(self) -> tt::Subtree {
        tt::Subtree { delimiter: None, token_trees: self }
    }

    fn to_tokens(self) -> Vec<tt::TokenTree> {
        self
    }
}

impl IntoTt for tt::Subtree {
    fn to_subtree(self) -> tt::Subtree {
        self
    }

    fn to_tokens(self) -> Vec<tt::TokenTree> {
        vec![tt::TokenTree::Subtree(self)]
    }
}

pub(crate) trait ToTokenTree {
    fn to_token(self) -> tt::TokenTree;
}

impl ToTokenTree for tt::TokenTree {
    fn to_token(self) -> tt::TokenTree {
        self
    }
}

impl ToTokenTree for tt::Subtree {
    fn to_token(self) -> tt::TokenTree {
        self.into()
    }
}

macro_rules! impl_to_to_tokentrees {
    ($($ty:ty => $this:ident $im:block);*) => {
        $(
            impl ToTokenTree for $ty {
                fn to_token($this) -> tt::TokenTree {
                    let leaf: tt::Leaf = $im.into();
                    leaf.into()
                }
            }

            impl ToTokenTree for &$ty {
                fn to_token($this) -> tt::TokenTree {
                    let leaf: tt::Leaf = $im.clone().into();
                    leaf.into()
                }
            }
        )*
    }
}

impl_to_to_tokentrees! {
    u32 => self { tt::Literal{text: self.to_string().into(), id: tt::TokenId::unspecified()} };
    usize => self { tt::Literal{text: self.to_string().into(), id: tt::TokenId::unspecified()} };
    i32 => self { tt::Literal{text: self.to_string().into(), id: tt::TokenId::unspecified()} };
    bool => self { tt::Ident{text: self.to_string().into(), id: tt::TokenId::unspecified()} };
    tt::Leaf => self { self };
    tt::Literal => self { self };
    tt::Ident => self { self };
    tt::Punct => self { self };
    &str => self { tt::Literal{text: format!("\"{}\"", self.escape_default()).into(), id: tt::TokenId::unspecified()}};
    String => self { tt::Literal{text: format!("\"{}\"", self.escape_default()).into(), id: tt::TokenId::unspecified()}}
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

    fn mk_ident(name: &str) -> tt::Ident {
        tt::Ident { text: name.into(), id: tt::TokenId::unspecified() }
    }

    #[test]
    fn test_quote_hash_token_tree() {
        let a = mk_ident("hello");

        let quoted = quote!(#a);
        assert_eq!(quoted.to_string(), "hello");
        let t = format!("{:?}", quoted);
        assert_eq!(t, "SUBTREE $\n  IDENT   hello 4294967295");
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

        let list = tt::Subtree {
            delimiter: Some(tt::Delimiter {
                kind: tt::DelimiterKind::Brace,
                id: tt::TokenId::unspecified(),
            }),
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
