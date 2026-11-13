// Regression test for #30225, which was an ICE that would trigger as
// a result of a poor interaction between trait result caching and
// type inference. Specifically, at that time, unification could cause
// unrelated type variables to become instantiated, if subtyping
// relationships existed. These relationships are now propagated
// through obligations and hence everything works out fine.

trait Foo<U,V> : Sized {
    fn foo(self, u: Option<U>, v: Option<V>) {}
}

struct A;
struct B;

impl Foo<A, B> for () {}      // impl A
impl Foo<u32, u32> for u32 {} // impl B, creating ambiguity

fn toxic() {
    // cache the resolution <() as Foo<$0,$1>> = impl A
    let u = None;
    let v = None;
    Foo::foo((), u, v);
}

fn bomb() {
    let mut u = None; // type is Option<$0>
    let mut v = None; // type is Option<$1>
    let mut x = None; // type is Option<$2>

    Foo::foo(x.unwrap(),u,v); // register <$2 as Foo<$0, $1>>
    u = v; // mark $0 and $1 in a subtype relationship
    //~^ ERROR mismatched types
    x = Some(()); // set $2 = (), allowing impl selection
                  // to proceed for <() as Foo<$0, $1>> = impl A.
                  // kaboom, this *used* to trigge an ICE
}

fn main() {}
