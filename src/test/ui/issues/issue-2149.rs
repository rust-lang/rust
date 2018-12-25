trait VecMonad<A> {
    fn bind<B, F>(&self, f: F) where F: FnMut(A) -> Vec<B>;
}

impl<A> VecMonad<A> for Vec<A> {
    fn bind<B, F>(&self, mut f: F) where F: FnMut(A) -> Vec<B> {
        let mut r = panic!();
        for elt in self { r = r + f(*elt); }
        //~^ ERROR E0277
   }
}
fn main() {
    ["hi"].bind(|x| [x] );
    //~^ ERROR no method named `bind` found for type `[&str; 1]` in the current scope
}
