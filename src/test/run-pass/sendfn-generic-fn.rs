// xfail-fast
#[legacy_modes];

extern mod std;

use comm::Chan;
use comm::send;

fn main() { test05(); }

type pair<A,B> = { a: A, b: B };

fn make_generic_record<A: Copy, B: Copy>(a: A, b: B) -> pair<A,B> {
    return {a: a, b: b};
}

fn test05_start(&&f: fn~(&&float, &&~str) -> pair<float, ~str>) {
    let p = f(22.22f, ~"Hi");
    log(debug, p);
    assert p.a == 22.22f;
    assert p.b == ~"Hi";

    let q = f(44.44f, ~"Ho");
    log(debug, q);
    assert q.a == 44.44f;
    assert q.b == ~"Ho";
}

fn spawn<A: Copy, B: Copy>(f: extern fn(fn~(A,B)->pair<A,B>)) {
    let arg = fn~(a: A, b: B) -> pair<A,B> {
        return make_generic_record(a, b);
    };
    task::spawn(|| f(arg) );
}

fn test05() {
    spawn::<float,~str>(test05_start);
}
