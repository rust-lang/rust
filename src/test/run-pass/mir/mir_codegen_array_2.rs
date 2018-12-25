// run-pass
fn into_inner(x: u64) -> [u64; 1024] {
    [x; 2*4*8*16]
}

fn main(){
    let x: &[u64] = &[42; 1024];
    assert_eq!(&into_inner(42)[..], x);
}
