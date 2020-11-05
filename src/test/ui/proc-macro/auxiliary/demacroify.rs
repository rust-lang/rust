// force-host
// no-prefer-dynamic

// An attr proc macro that removes all postfix macros,
// to test that parsing postfix macros is allowed.

#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{Delimiter, Group, Punct, Spacing, TokenStream, TokenTree as Tt};

#[proc_macro_attribute]
pub fn demacroify(_attrs: TokenStream, input: TokenStream) -> TokenStream {
    let mut vis = Visitor;
    let res = vis.visit_stream(input);
    res
}

struct Visitor;

impl Visitor {
    fn visit_stream(&mut self, stream: TokenStream) -> TokenStream {
        let mut res = Vec::new();
        let mut stream_iter = stream.into_iter();
        while let Some(tt) = stream_iter.next() {
            match tt {
                Tt::Group(group) => {
                    let mut postfix_macro = false;
                    {
                        let last_three = res.rchunks(3).next();
                        if let Some(&[Tt::Punct(ref p1), Tt::Ident(_), Tt::Punct(ref p2)]) =
                            last_three
                        {
                            if (p1.as_char(), p1.spacing(), p2.as_char(), p2.spacing())
                                == ('.', Spacing::Alone, '!', Spacing::Alone)
                            {
                                postfix_macro = true;
                            }
                        }
                    }
                    if postfix_macro {
                        // Remove the ! and macro ident
                        let _mac_bang = res.pop().unwrap();
                        let _mac = res.pop().unwrap();
                        // Remove the . before the macro
                        let _dot = res.pop().unwrap();
                    } else {
                        let tt = Tt::Group(self.visit_group(group));
                        res.push(tt);
                    }
                }
                Tt::Ident(id) => {
                    res.push(Tt::Ident(id));
                }
                Tt::Punct(p) => {
                    res.push(Tt::Punct(p));
                }
                Tt::Literal(lit) => {
                    res.push(Tt::Literal(lit));
                }
            }
        }
        res.into_iter().collect()
    }
    fn visit_group(&mut self, group: Group) -> Group {
        let delim = group.delimiter();
        let span = group.span();
        let stream = self.visit_stream(group.stream());
        let mut gr = Group::new(delim, stream);
        gr.set_span(span);
        gr
    }
}
