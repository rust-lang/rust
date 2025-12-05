//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

trait A<'a> {}
trait B<'b> {}
fn foo<T>()
where
    for<'a> T: A<'a> + 'a,
{
}
trait C<'c>: for<'a> A<'a> + for<'b> B<'b> {
    type As;
}
struct D<T>
where
    T: for<'c> C<'c, As: A<'c>>,
{
    t: std::marker::PhantomData<T>,
}

trait E<'e> {
    type As;
}
trait F<'f>: for<'a> A<'a> + for<'e> E<'e> {}
struct G<T>
where
    for<'f> T: F<'f, As: E<'f>> + 'f,
{
    t: std::marker::PhantomData<T>,
}

trait I<'a, 'b, 'c> {
    type As;
}
trait H<'d, 'e>: for<'f> I<'d, 'f, 'e> + 'd {}
fn foo2<T>()
where
    T: for<'g> H<'g, 'g, As: for<'h> H<'h, 'g> + 'g>,
{
}

fn foo3<T>()
where
    T: for<'i> H<'i, 'i, As: for<'j> H<'j, 'i, As: for<'k> I<'i, 'k, 'j> + 'j> + 'i>,
{
}
fn foo4<T>()
where
    T: for<'l, 'i> H<'l, 'i, As: for<'j> H<'j, 'i, As: for<'k> I<'l, 'k, 'j> + 'j> + 'i>,
{
}

struct X<'x, 'y> {
    x: std::marker::PhantomData<&'x ()>,
    y: std::marker::PhantomData<&'y ()>,
}

fn foo5<T>()
where
    T: for<'l, 'i> H<'l, 'i, As: for<'j> H<'j, 'i, As: for<'k> H<'j, 'k, As = X<'j, 'k>> + 'j> + 'i>
{
}

fn main() {}
