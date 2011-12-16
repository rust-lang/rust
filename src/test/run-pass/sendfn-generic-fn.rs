use std;

import std::comm;
import std::comm::chan;
import std::comm::send;

fn main() { test05(); }

type pair<A,B> = { a: A, b: B };

fn make_generic_record<copy A,copy B>(a: A, b: B) -> pair<A,B> {
    ret {a: a, b: b};
}

fn test05_start(&&f: sendfn(&&float, &&str) -> pair<float, str>) {
    let p = f(22.22f, "Hi");
    log p;
    assert p.a == 22.22f;
    assert p.b == "Hi";

    let q = f(44.44f, "Ho");
    log q;
    assert q.a == 44.44f;
    assert q.b == "Ho";
}

fn spawn<copy A, copy B>(f: fn(sendfn(A,B)->pair<A,B>)) {
    let arg = sendfn(a: A, b: B) -> pair<A,B> {
        ret make_generic_record(a, b);
    };
    task::spawn(arg, f);
}

fn test05() {
    spawn::<float,str>(test05_start);
}
