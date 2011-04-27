// xfail-boot
use std;
import std._vec.len;
fn main() {

    // Pseudomethod using a library function
    let uint a = len[int](vec(1, 2, 3, 4));
    log a;
    let uint b = vec(1, 2, 3, 4)::len[int]();
    log b;
    check (a == b);

    let vec[int] v = vec(1, 2, 3, 4);
    let uint c = len[int](v);
    log c;
    let uint d = v::len[int]();
    log d;
    check (c == d);

    // User-defined pseudomethods
    fn exclaim(str s) -> str { ret s + "!"; }
    let str e = exclaim("hello");
    log e;
    let str f = "hello"::exclaim();
    log f;
    check (e == f);

    fn plus(int a, int b) -> int { ret a + b; }
    let int m = 2 * 3::plus(4) + 5;
    let int n = 2 * (3 + 4) + 5;
    log m;
    log n;
    check (m == n);

    // Multi-argument pseudomethod
    fn bang_huh(str s1, str s2) -> str { 
        ret s1 + "!" + s2 + "?";
    }
    let str g = bang_huh("hello", "world");
    log g;
    let str h = "hello"::bang_huh("world");
    log h;
    check (g == h);

    // Stacking pseudomethods
    let str i = "hello"::exclaim()::bang_huh("world");
    log i;
    let str j = bang_huh(exclaim("hello"), "world");
    log j;
    check (i == j);

    let int k = (vec("foo", "bar", "baz")::len[str]() as int)::plus(50);
    log k;
    check (k == 53);
}
