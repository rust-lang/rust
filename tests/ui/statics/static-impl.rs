//@ run-pass
#![allow(non_camel_case_types)]




pub trait plus {
    fn plus(&self) -> isize;
}

mod a {
    use crate::plus;
    impl plus for usize { fn plus(&self) -> isize { *self as isize + 20 } }
}

mod b {
    use crate::plus;
    impl plus for String { fn plus(&self) -> isize { 200 } }
}

trait uint_utils {
    fn str(&self) -> String;
    fn multi<F>(&self, f: F) where F: FnMut(usize);
}

impl uint_utils for usize {
    fn str(&self) -> String {
        self.to_string()
    }
    fn multi<F>(&self, mut f: F) where F: FnMut(usize) {
        let mut c = 0_usize;
        while c < *self { f(c); c += 1_usize; }
    }
}

trait vec_utils<T> {
    fn length_(&self, ) -> usize;
    fn iter_<F>(&self, f: F) where F: FnMut(&T); //~ WARN method `iter_` is never used
    fn map_<U, F>(&self, f: F) -> Vec<U> where F: FnMut(&T) -> U;
}

impl<T> vec_utils<T> for Vec<T> {
    fn length_(&self) -> usize { self.len() }
    fn iter_<F>(&self, mut f: F) where F: FnMut(&T) { for x in self { f(x); } }
    fn map_<U, F>(&self, mut f: F) -> Vec<U> where F: FnMut(&T) -> U {
        let mut r = Vec::new();
        for elt in self {
            r.push(f(elt));
        }
        r
    }
}

pub fn main() {
    assert_eq!(10_usize.plus(), 30);
    assert_eq!(("hi".to_string()).plus(), 200);

    assert_eq!((vec![1]).length_().str(), "1".to_string());
    let vect = vec![3, 4].map_(|a| *a + 4);
    assert_eq!(vect[0], 7);
    let vect = (vec![3, 4]).map_::<usize, _>(|a| *a as usize + 4_usize);
    assert_eq!(vect[0], 7_usize);
    let mut x = 0_usize;
    10_usize.multi(|_n| x += 2_usize );
    assert_eq!(x, 20_usize);
}
