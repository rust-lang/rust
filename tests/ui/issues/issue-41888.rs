//@ run-pass
fn main() { let _ = g(Some(E::F(K))); }

type R = Result<(), ()>;
struct K;

enum E {
    F(K), // must not be built-in type
    #[allow(dead_code)]
    G(Box<E>, Box<E>),
}

fn translate(x: R) -> R { x }

fn g(mut status: Option<E>) -> R {
    loop {
        match status {
            Some(infix_or_postfix) => match infix_or_postfix {
                E::F(_op) => { // <- must be captured by value
                    match Ok(()) {
                        Err(err) => return Err(err),
                        Ok(_) => {},
                    };
                }
                _ => (),
            },
            _ => match translate(Err(())) {
                Err(err) => return Err(err),
                Ok(_) => {},
            }
        }
        status = None;
    }
}
