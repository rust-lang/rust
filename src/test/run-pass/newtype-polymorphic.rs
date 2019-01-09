#![allow(non_camel_case_types)]


#[derive(Clone)]
struct myvec<X>(Vec<X> );

fn myvec_deref<X:Clone>(mv: myvec<X>) -> Vec<X> {
    let myvec(v) = mv;
    return v.clone();
}

fn myvec_elt<X>(mv: myvec<X>) -> X {
    let myvec(v) = mv;
    return v.into_iter().next().unwrap();
}

pub fn main() {
    let mv = myvec(vec![1, 2, 3]);
    let mv_clone = mv.clone();
    let mv_clone = myvec_deref(mv_clone);
    assert_eq!(mv_clone[1], 2);
    assert_eq!(myvec_elt(mv.clone()), 1);
    let myvec(v) = mv;
    assert_eq!(v[2], 3);
}
