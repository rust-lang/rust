// xfail-stage0

fn main()
{
    // Make sure we properly handle repeated self-appends.
    let vec[int] a = [0];
    auto i = 20;
    while (i > 0) {
        a += a;
        i -= 1;
    }
}
