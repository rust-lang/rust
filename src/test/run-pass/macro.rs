// xfail-pretty - token trees can't pretty print

fn main() {
    #macro[[#m1[a], a * 4]];
    assert (m1!{2} == 8);

    macro_rules! m1tt {
        {$a:expr} => {$a*4}
    };
    assert(m1tt!{2} == 8);
}
