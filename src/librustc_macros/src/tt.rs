use proc_macro2::{Delimiter, TokenStream, TokenTree};

pub struct TT<'a>(pub &'a TokenTree);

impl<'a> PartialEq for TT<'a> {
    fn eq(&self, other: &Self) -> bool {
        use proc_macro2::Spacing;

        match (self.0, other.0) {
            (&TokenTree::Group(ref g1), &TokenTree::Group(ref g2)) => {
                match (g1.delimiter(), g2.delimiter()) {
                    (Delimiter::Parenthesis, Delimiter::Parenthesis)
                    | (Delimiter::Brace, Delimiter::Brace)
                    | (Delimiter::Bracket, Delimiter::Bracket)
                    | (Delimiter::None, Delimiter::None) => {}
                    _ => return false,
                }

                let s1 = g1.stream().clone().into_iter();
                let mut s2 = g2.stream().clone().into_iter();

                for item1 in s1 {
                    let item2 = match s2.next() {
                        Some(item) => item,
                        None => return false,
                    };
                    if TT(&item1) != TT(&item2) {
                        return false;
                    }
                }
                s2.next().is_none()
            }
            (&TokenTree::Punct(ref o1), &TokenTree::Punct(ref o2)) => {
                o1.as_char() == o2.as_char()
                    && match (o1.spacing(), o2.spacing()) {
                        (Spacing::Alone, Spacing::Alone) | (Spacing::Joint, Spacing::Joint) => true,
                        _ => false,
                    }
            }
            (&TokenTree::Literal(ref l1), &TokenTree::Literal(ref l2)) => {
                l1.to_string() == l2.to_string()
            }
            (&TokenTree::Ident(ref s1), &TokenTree::Ident(ref s2)) => s1 == s2,
            _ => false,
        }
    }
}

pub struct TS<'a>(pub &'a TokenStream);

impl<'a> PartialEq for TS<'a> {
    fn eq(&self, other: &Self) -> bool {
        let left = self.0.clone().into_iter().collect::<Vec<_>>();
        let right = other.0.clone().into_iter().collect::<Vec<_>>();
        if left.len() != right.len() {
            return false;
        }
        for (a, b) in left.into_iter().zip(right) {
            if TT(&a) != TT(&b) {
                return false;
            }
        }
        true
    }
}
