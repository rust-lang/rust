// xfail-pretty - token trees can't pretty print

fn main() {
    #macro[[#trivial[], 1 * 2 * 4 * 2 * 1]];

    assert (trivial!{} == 16);

    macro_rules! trivial_tt{
        {} => {1*2*4*2*1}
    }
    assert(trivial_tt!{} == 16);
}
