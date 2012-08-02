fn fail_len(v: ~[const int]) -> uint {
    let mut i = fail;
    for v.each |x| { i += 1u; }
    //~^ WARNING unreachable statement
    //~^^ ERROR the type of this value must be known
    return i;
}
fn main() {}