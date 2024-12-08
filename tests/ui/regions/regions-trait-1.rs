//@ check-pass

struct Ctxt {
    v: usize,
}

trait GetCtxt {
    // Here the `&` is bound in the method definition:
    fn get_ctxt(&self) -> &Ctxt;
}

struct HasCtxt<'a> {
    c: &'a Ctxt,
}

impl<'a> GetCtxt for HasCtxt<'a> {
    // Ok: Have implied bound of WF(&'b HasCtxt<'a>)
    // so know 'a: 'b
    // so know &'a Ctxt <: &'b Ctxt
    fn get_ctxt<'b>(&'b self) -> &'a Ctxt {
        self.c
    }
}

fn get_v(gc: Box<dyn GetCtxt + '_>) -> usize {
    gc.get_ctxt().v
}

fn main() {
    let ctxt = Ctxt { v: 22 };
    let hc = HasCtxt { c: &ctxt };
    assert_eq!(get_v(Box::new(hc) as Box<dyn GetCtxt>), 22);
}
