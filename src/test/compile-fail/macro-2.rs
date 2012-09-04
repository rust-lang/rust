//error-pattern:is an expr, expected an identifier
fn main() {
    #macro[[#mylambda[x, body],
            {
                fn f(x: int) -> int { return body }
                f
            }]];

    assert (mylambda!(y * 1, y * 2)(8) == 16);
}
