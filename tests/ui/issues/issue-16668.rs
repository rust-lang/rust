//@ check-pass
#![allow(dead_code)]
struct Parser<'a, I, O> {
    parse: Box<dyn FnMut(I) -> Result<O, String> + 'a>
}

impl<'a, I: 'a, O: 'a> Parser<'a, I, O> {
    fn compose<K: 'a>(mut self, mut rhs: Parser<'a, O, K>) -> Parser<'a, I, K> {
        Parser {
            parse: Box::new(move |x: I| {
                match (self.parse)(x) {
                    Ok(r) => (rhs.parse)(r),
                    Err(e) => Err(e)
                }
            })
        }
    }
}

fn main() {}
