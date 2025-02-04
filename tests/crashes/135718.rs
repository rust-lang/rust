//@ known-bug: #135718

struct Equal;

struct Bar;

trait TwiceNested {}
impl<M> TwiceNested for Bar where Bar: NestMakeEqual<NestEq = M> {}

struct Sum;

trait Not {
    fn not();
}

impl<P> Not for Sum
where
    Bar: NestMakeEqual<NestEq = P>,
    Self: Problem<P>,
{
    fn not() {}
}

trait NestMakeEqual {
    type NestEq;
}

trait MakeEqual {
    type Eq;
}

struct Foo;
impl MakeEqual for Foo {
    type Eq = Equal;
}

impl<O> NestMakeEqual for Bar
where
    Foo: MakeEqual<Eq = O>,
{
    type NestEq = O;
}

trait Problem<M> {}
impl Problem<()> for Sum where Bar: TwiceNested {}
impl Problem<Equal> for Sum where Bar: TwiceNested {}

fn main() {
    Sum::not();
}
