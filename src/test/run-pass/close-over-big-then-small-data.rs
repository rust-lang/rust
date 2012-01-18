// If we use GEPi rathern than GEP_tup_like when
// storing closure data (as we used to do), the u64 would
// overwrite the u16.

type pair<A,B> = {
    a: A, b: B
};

fn f<A:copy>(a: A, b: u16) -> fn@() -> (A, u16) {
    fn@() -> (A, u16) { (a, b) }
}

fn main() {
    let (a, b) = f(22_u64, 44u16)();
    #debug["a=%? b=%?", a, b];
    assert a == 22u64;
    assert b == 44u16;
}
