// xfail-pretty - token trees can't pretty print

fn main() {
    #macro[[#mylambda[x, body],
            {
                fn f(x: int) -> int { ret body; }
                f
            }]];

    assert (mylambda!{y, y * 2}(8) == 16);

    macro_rules! mylambda_tt{
        {$x:ident, $body:expr} => {
            fn f($x: int) -> int { ret $body; };
            f
        }
    }

    assert(mylambda_tt!{y, y * 2}(8) == 16)
}
