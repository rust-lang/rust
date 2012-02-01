

// Regression test for issue #374
use std;
import option;
import option::none;

enum sty { ty_nil, }

type raw_t = {struct: sty, cname: option<str>, hash: uint};

fn mk_raw_ty(st: sty, cname: option<str>) -> raw_t {
    ret {struct: st, cname: cname, hash: 0u};
}

fn main() { mk_raw_ty(ty_nil, none::<str>); }
