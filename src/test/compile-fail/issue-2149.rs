impl monad<A> for ~[A] {
    fn bind<B>(f: fn(A) -> ~[B]) {
        let mut r = fail;
        for self.each |elt| { r += f(elt); }
        //~^ WARNING unreachable expression
        //~^^ ERROR the type of this value must be known
   }
}
fn main() {
    ["hi"].bind({|x| [x] });
}