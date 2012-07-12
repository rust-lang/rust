iface get_ctxt {
    fn get_ctxt() -> &self/uint;
}

fn make_gc1(gc: get_ctxt/&a) -> get_ctxt/&b  {
    ret gc; //~ ERROR mismatched types: expected `get_ctxt/&b` but found `get_ctxt/&a`
}

fn make_gc2(gc: get_ctxt/&a) -> get_ctxt/&b  {
    ret gc as get_ctxt; //~ ERROR mismatched types: expected `get_ctxt/&b` but found `get_ctxt/&a`
}

fn main() {
}
