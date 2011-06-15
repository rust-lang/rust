

// Regression test for issue #374
use std;
import std::option;
import std::option::none;

tag sty { ty_nil; }

type raw_t = rec(sty struct, option::t[str] cname, uint hash);

fn mk_raw_ty(sty st, &option::t[str] cname) -> raw_t {
    ret rec(struct=st, cname=cname, hash=0u);
}

fn main() { mk_raw_ty(ty_nil, none[str]); }