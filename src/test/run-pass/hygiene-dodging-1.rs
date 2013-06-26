mod x {
    pub fn g() -> uint {14}
}

fn main(){
    // should *not* shadow the module x:
    let x = 9;
    // use it to avoid warnings:
    x+3;
    assert_eq!(x::g(),14);
}
