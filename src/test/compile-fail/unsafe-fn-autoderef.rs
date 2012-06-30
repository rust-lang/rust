// -*- rust -*-

type rec = {f: int};
fn f(p: *rec) -> int {

    // Test that * ptrs do not autoderef.  There is a deeper reason for
    // prohibiting this, beyond making unsafe things annoying (which doesn't
    // actually seem desirable to me).  The deeper reason is that if you
    // have a type like:
    //
    //    enum foo = *foo;
    //
    // you end up with an infinite auto-deref chain, which is
    // currently impossible (in all other cases, infinite auto-derefs
    // are prohibited by various checks, such as that the enum is
    // instantiable and so forth).

    ret p.f; //~ ERROR attempted access of field `f` on type `*rec`
}

fn main() {
}
