type alist<A,B> = { eq_fn: fn@(A,A) -> bool, mut data: [(A,B)] };

fn alist_add<A: copy, B: copy>(lst: alist<A,B>, k: A, v: B) {
    lst.data += [(k, v)];
}

fn alist_get<A: copy, B: copy>(lst: alist<A,B>, k: A) -> B {
    let eq_fn = lst.eq_fn;
    for pair in lst.data {
        let (ki, vi) = pair; // copy req'd for alias analysis
        if eq_fn(k, ki) { ret vi; }
    }
    fail;
}

#[inline]
fn new_int_alist<B: copy>() -> alist<int, B> {
    fn eq_int(&&a: int, &&b: int) -> bool { a == b }
    ret {eq_fn: eq_int,
         mut data: []};
}

#[inline]
fn new_int_alist_2<B: copy>() -> alist<int, B> {
    #[inline]
    fn eq_int(&&a: int, &&b: int) -> bool { a == b }
    ret {eq_fn: eq_int,
         mut data: []};
}