// Tests different fns

fn foo(a: AAAA, b: BBB, c: CCC) -> RetType {}

fn foo(a: AAAA, b: BBB /* some, weird, inline comment */, c: CCC) -> RetType
    where T: Blah
{

}

fn foo(a: AAA /* (comment) */)
    where T: Blah
{

}

fn foo(a: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
       b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
       -> RetType
    where T: Blah
{

}


fn foo<U, T>(a: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,
             b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
             -> RetType
    where T: Blah,
          U: dsfasdfasdfasd
{

}

fn foo<U: Fn(A) -> B /* paren inside generics */>() {}

impl Foo {
    fn with_no_errors<T, F>(&mut self, f: F) -> T
        where F: FnOnce(&mut Resolver) -> T
    {
    }

    fn foo(mut self, mut bar: u32) {}

    fn bar(self, mut bazz: u32) {}
}

pub fn render<'a,
              N: Clone + 'a,
              E: Clone + 'a,
              G: Labeller<'a, N, E> + GraphWalk<'a, N, E>,
              W: Write>
    (g: &'a G,
     w: &mut W)
     -> io::Result<()> {
    render_opts(g, w, &[])
}

const fn foo() {
    x;
}

pub const fn foo() {
    x;
}

impl Foo {
    const fn foo() {
        x;
    }
}

fn homura<T: Deref<Target = i32>>(_: T) {}

fn issue377() -> (Box<CompositorProxy + Send>, Box<CompositorReceiver>) {}

fn main() {
    let _ = function(move || 5);
    let _ = move || 42;
    let _ = || unsafe { abort() };
}

// With inner attributes.
fn inner() {
    #![inline]
    x
}
