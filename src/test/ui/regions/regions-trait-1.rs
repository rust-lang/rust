#![feature(box_syntax)]

struct Ctxt { v: usize }

trait GetCtxt {
    // Here the `&` is bound in the method definition:
    fn get_ctxt(&self) -> &Ctxt;
}

struct HasCtxt<'a> { c: &'a Ctxt }

impl<'a> GetCtxt for HasCtxt<'a> {

    // Here an error occurs because we used `&self` but
    // the definition used `&`:
    fn get_ctxt(&self) -> &'a Ctxt { //~ ERROR method not compatible with trait
        self.c
    }

}

fn get_v(gc: Box<dyn GetCtxt>) -> usize {
    gc.get_ctxt().v
}

fn main() {
    let ctxt = Ctxt { v: 22 };
    let hc = HasCtxt { c: &ctxt };
    assert_eq!(get_v(box hc as Box<dyn GetCtxt>), 22);
}
