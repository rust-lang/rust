type ctxt = { v: uint };

trait get_ctxt {
    // Here the `&` is bound in the method definition:
    fn get_ctxt() -> &ctxt;
}

type has_ctxt = { c: &ctxt };

impl has_ctxt: get_ctxt {

    // Here an error occurs because we used `&self` but
    // the definition used `&`:
    fn get_ctxt() -> &self/ctxt { //~ ERROR method `get_ctxt` has an incompatible type
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
