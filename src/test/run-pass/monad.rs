#[legacy_modes];

trait vec_monad<A> {
    fn bind<B: Copy>(f: fn(A) -> ~[B]) -> ~[B];
}

impl<A> ~[A]: vec_monad<A> {
    fn bind<B: Copy>(f: fn(A) -> ~[B]) -> ~[B] {
        let mut r = ~[];
        for self.each |elt| { r += f(elt); }
        r
    }
}

trait option_monad<A> {
    fn bind<B>(f: fn(A) -> Option<B>) -> Option<B>;
}

impl<A> Option<A>: option_monad<A> {
    fn bind<B>(f: fn(A) -> Option<B>) -> Option<B> {
        match self {
          Some(a) => { f(a) }
          None => { None }
        }
    }
}

fn transform(x: Option<int>) -> Option<~str> {
    x.bind(|n| Some(n + 1) ).bind(|n| Some(int::str(n)) )
}

fn main() {
    assert transform(Some(10)) == Some(~"11");
    assert transform(None) == None;
    assert (~[~"hi"]).bind(|x| ~[x, x + ~"!"] ).bind(|x| ~[x, x + ~"?"] ) ==
        ~[~"hi", ~"hi?", ~"hi!", ~"hi!?"];
}
