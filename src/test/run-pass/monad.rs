trait vec_monad<A> {
    fn bind<B>(f: fn(A) -> ~[B]) -> ~[B];
}

impl<A> ~[A]: vec_monad<A> {
    fn bind<B>(f: fn(A) -> ~[B]) -> ~[B] {
        let mut r = ~[];
        for self.each |elt| { r += f(elt); }
        r
    }
}

trait option_monad<A> {
    fn bind<B>(f: fn(A) -> option<B>) -> option<B>;
}

impl<A> option<A>: option_monad<A> {
    fn bind<B>(f: fn(A) -> option<B>) -> option<B> {
        match self {
          some(a) => { f(a) }
          none => { none }
        }
    }
}

fn transform(x: option<int>) -> option<~str> {
    x.bind(|n| some(n + 1) ).bind(|n| some(int::str(n)) )
}

fn main() {
    assert transform(some(10)) == some(~"11");
    assert transform(none) == none;
    assert (~[~"hi"]).bind(|x| ~[x, x + ~"!"] ).bind(|x| ~[x, x + ~"?"] ) ==
        ~[~"hi", ~"hi?", ~"hi!", ~"hi!?"];
}
