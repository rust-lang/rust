type ctxt = { v: uint };

iface get_ctxt {
    fn get_ctxt() -> &self/ctxt;
}

type has_ctxt = { c: &ctxt };

impl of get_ctxt for has_ctxt {
    fn get_ctxt() -> &self/ctxt { self.c }
}

fn make_gc() -> get_ctxt  {
    let ctxt = { v: 22u };
    let hc = { c: &ctxt };
    ret hc as get_ctxt; //~ ERROR mismatched types: expected `get_ctxt/&`
}

fn main() {
    make_gc().get_ctxt().v;
}
