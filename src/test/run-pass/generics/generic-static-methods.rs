// run-pass
#![allow(non_camel_case_types)]


trait vec_utils<T> {
    fn map_<U, F>(x: &Self, f: F) -> Vec<U> where F: FnMut(&T) -> U;
}

impl<T> vec_utils<T> for Vec<T> {
    fn map_<U, F>(x: &Vec<T> , mut f: F) -> Vec<U> where F: FnMut(&T) -> U {
        let mut r = Vec::new();
        for elt in x {
            r.push(f(elt));
        }
        r
    }
}

pub fn main() {
    assert_eq!(vec_utils::map_(&vec![1,2,3], |&x| x+1), [2,3,4]);
}
