// xfail-boot
use std;
import std._vec.len;
fn main() {

    // Sanity check
    let uint a = len[int](vec(1, 2, 3, 4));
    log a;
    let vec[int] v = vec(1, 2, 3, 4);
    let uint b = len[int](v);
    log b;
    check (a == b);

    // Pseudomethod
    let uint c = vec(1, 2, 3, 4)::len[int]();
    log c;
    let uint d = v::len[int]();
    log d;
    check (c == d);

    // User-defined pseudomethod
    fn exclaim(str s) -> str { ret s + "!"; }
    let str e = exclaim("hello");
    log e;
    let str f = "hello"::exclaim();
    log f;
    check (e == f);
}
