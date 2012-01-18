type pair<A,B> = {
    a: A, b: B
};

fn f<A:send,B:send>(a: A, b: B) -> fn~() -> (A, B) {
    fn~() -> (A, B) { (a, b) }
}

fn main() {
    let x = 22_u8;
    let y = 44_u64;
    let (a, b) = f(~x, ~y)();
    #debug["a=%? b=%?", *a, *b];
    assert *a == x;
    assert *b == y;
}