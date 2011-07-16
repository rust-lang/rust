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

   To fix this bug, we need to make the vtable slots on the inner
   object match whatever the object being passed in at runtime has.
   My first instinct was to change the vtable to match the runtime
   object, but vtables are already baked into RO memory.  So, instead,
   we're going to tweak the object being passed in at runtime to match
   the vtable that inner already has.  That is, it needs to only have
   a, m, and z slots in its vtable, and each one of those slots will
   forward to the *outer* object's a, m, and z slots, respectively.
   From there they will either head right back to inner, or they'll be
   overridden.

   Adding support for this is issue #702.

*/
