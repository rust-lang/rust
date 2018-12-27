#![allow(non_camel_case_types)]
#![allow(dead_code)]



// Regression test for issue #374

// pretty-expanded FIXME #23616

enum sty { ty_nil, }

struct RawT {struct_: sty, cname: Option<String>, hash: usize}

fn mk_raw_ty(st: sty, cname: Option<String>) -> RawT {
    return RawT {struct_: st, cname: cname, hash: 0};
}

pub fn main() { mk_raw_ty(sty::ty_nil, None::<String>); }
