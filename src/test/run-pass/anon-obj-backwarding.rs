//xfail-stage0
//xfail-stage1
//xfail-stage2
use std;

fn main() {

    obj inner() {
        fn a() -> int {
            ret 2;
        }
        fn m() -> uint {
            ret 3u;
        }
        fn z() -> uint {
            ret self.m();
        }
    }

    auto my_inner = inner();

    auto my_outer = obj() {
        fn b() -> uint {
            ret 5u;
        }
        fn n() -> str {
            ret "world!";
        }
        with my_inner
    };

    log_err my_inner.z();
    assert (my_inner.z() == 3u);
    log_err my_outer.z();
    assert (my_outer.z() == 3u);
}

/*
   Here, when we make the self-call to self.m() in inner, we're going
   back through the outer "self".  That outer "self" has 5 methods in
   its vtable: a, b, m, n, z.  But the method z has already been
   compiled, and at the time it was compiled, it expected "self" to
   only have three methods in its vtable: a, m, and z.  So, the method
   z thinks that "self.m()" means "look up method #1 (indexing from 0)
   in my vtable and call it".  That means that it'll call method #1 on
   the larger vtable that it thinks is "self", and method #1 at that
   point is b.

   So, when we call my_inner.z(), we get 3, which is what we'd
   expect.  When we call my_outer.z(), we should also get 3, because
   at no point is z being overridden.

   To fix this bug, we need to add a second level of forwarding
   functions (let's call them "backwarding functions") on the inner
   object.  Every time an object is extended with another object, we
   have to rewrite the inner object's vtable to account for the fact
   that future self-calls will get a larger object.  The inner
   object's vtable will need to have five slots, too.  The ones for b
   and n will point right back at the outer object.  (These are the
   "backwarding" ones.)  And the ones for a, m, and z will point at
   the original, real vtable for inner.

   Adding support for this is issue #702.

*/
