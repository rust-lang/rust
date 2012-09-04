type ctxt = { v: uint };

trait get_ctxt {
    fn get_ctxt() -> &self/ctxt;
}

type has_ctxt = { c: &ctxt };

impl has_ctxt: get_ctxt {
    fn get_ctxt() -> &self/ctxt {
        self.c
    }
}

fn get_v(gc: get_ctxt) -> uint {
    gc.get_ctxt().v
}

fn main() {
    let ctxt = { v: 22u };
    let hc = { c: &ctxt };

    assert get_v(hc as get_ctxt) == 22u;
}
