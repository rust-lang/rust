// Test for the outlives relation when applied to a projection on a
// type with bound regions. In this case, we are checking that
// `<for<'r> fn(&'r T) as TheTrait>::TheType: 'a` If we're not
// careful, we could wind up with a constraint that `'r:'a`, but since
// `'r` is bound, that leads to badness. This test checks that
// everything works.

//@ check-pass
#![allow(dead_code)]

trait TheTrait {
    type TheType;
}

fn wf<T>() { }

type FnType<T> = for<'r> fn(&'r T);

fn foo<'a,'b,T>()
    where FnType<T>: TheTrait
{
    wf::< <FnType<T> as TheTrait>::TheType >();
}


fn main() { }
