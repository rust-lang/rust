// Tests different fns

fn foo(a: AAAA, b: BBB, c: CCC) -> RetType {

}

fn foo(a: AAAA, b: BBB, c: CCC) -> RetType
    where T: Blah
{

}

fn foo(a: AAA)
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

impl Foo {
    fn with_no_errors<T, F>(&mut self, f: F) -> T
        where F: FnOnce(&mut Resolver) -> T
    {
    }
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
