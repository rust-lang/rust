// run-pass

#![allow(non_camel_case_types)]



trait vec_monad<A> {
    fn bind<B, F>(&self, f: F ) -> Vec<B> where F: FnMut(&A) -> Vec<B> ;
}

impl<A> vec_monad<A> for Vec<A> {
    fn bind<B, F>(&self, mut f: F) -> Vec<B> where F: FnMut(&A) -> Vec<B> {
        let mut r = Vec::new();
        for elt in self {
            r.extend(f(elt));
        }
        r
    }
}

trait option_monad<A> {
    fn bind<B, F>(&self, f: F) -> Option<B> where F: FnOnce(&A) -> Option<B>;
}

impl<A> option_monad<A> for Option<A> {
    fn bind<B, F>(&self, f: F) -> Option<B> where F: FnOnce(&A) -> Option<B> {
        match *self {
            Some(ref a) => { f(a) }
            None => { None }
        }
    }
}

fn transform(x: Option<isize>) -> Option<String> {
    x.bind(|n| Some(*n + 1) ).bind(|n| Some(n.to_string()) )
}

pub fn main() {
    assert_eq!(transform(Some(10)), Some("11".to_string()));
    assert_eq!(transform(None), None);
    assert_eq!((vec!["hi".to_string()])
        .bind(|x| vec![x.clone(), format!("{}!", x)] )
        .bind(|x| vec![x.clone(), format!("{}?", x)] ),
        ["hi".to_string(),
         "hi?".to_string(),
         "hi!".to_string(),
         "hi!?".to_string()]);
}
