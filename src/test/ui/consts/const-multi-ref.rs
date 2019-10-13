const _X: i32 = {
    let mut a = 5;
    let p = &mut a;      //~ ERROR references in constants may only refer to immutable values

    let reborrow = {p};  //~ ERROR references in constants may only refer to immutable values
    let pp = &reborrow;
    let ppp = &pp;
    ***ppp
};

fn main() {}
