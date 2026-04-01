extern crate proc_macro;
use proc_macro::{Ident, Group, TokenStream, TokenTree as Tt};

// This constant has to be above the ALLOCATING_ALGO_THRESHOLD
// constant in inherent_impls_overlap.rs
const REPEAT_COUNT: u32 = 501;

#[proc_macro]
/// Repeats the input many times, while replacing idents
/// named "IDENT" with "id_$v", where v is a counter.
pub fn repeat_with_idents(input: TokenStream) -> TokenStream {
    let mut res = Vec::new();
    fn visit_stream(res: &mut Vec<Tt>, stream :TokenStream, v: u32) {
        let mut stream_iter = stream.into_iter();
        while let Some(tt) = stream_iter.next() {
            match tt {
                Tt::Group(group) => {
                    let tt = Tt::Group(visit_group(group, v));
                    res.push(tt);
                },
                Tt::Ident(id) => {
                    let id = if &id.to_string() == "IDENT" {
                        Ident::new(&format!("id_{}", v), id.span())
                    } else {
                        id
                    };
                    res.push(Tt::Ident(id));
                },
                Tt::Punct(p) => {
                    res.push(Tt::Punct(p));
                },
                Tt::Literal(lit) => {
                    res.push(Tt::Literal(lit));
                },
            }
        }
    }
    fn visit_group(group :Group, v: u32) -> Group {
        let mut res = Vec::new();
        visit_stream(&mut res, group.stream(), v);
        let stream = res.into_iter().collect();
        let delim = group.delimiter();
        Group::new(delim, stream)
    }
    for v in 0 .. REPEAT_COUNT {
        visit_stream(&mut res, input.clone(), v)
    }
    res.into_iter().collect()
}
