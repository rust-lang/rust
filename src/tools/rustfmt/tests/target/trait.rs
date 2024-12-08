// Test traits

trait Foo {
    fn bar(x: i32) -> Baz<U> {
        Baz::new()
    }

    fn baz(a: AAAAAAAAAAAAAAAAAAAAAA, b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB) -> RetType;

    fn foo(
        a: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, // Another comment
        b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,
    ) -> RetType; // Some comment

    fn baz(&mut self) -> i32;

    fn increment(&mut self, x: i32);

    fn read(&mut self, x: BufReader<R> /* Used to be MemReader */)
    where
        R: Read;
}

pub trait WriteMessage {
    fn write_message(&mut self, &FrontendMessage) -> io::Result<()>;
}

trait Runnable {
    fn handler(self: &Runnable);
}

trait TraitWithExpr {
    fn fn_with_expr(x: [i32; 1]);
}

trait Test {
    fn read_struct<T, F>(&mut self, s_name: &str, len: usize, f: F) -> Result<T, Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<T, Self::Error>;
}

trait T {}

trait Foo {
    type Bar: Baz;
    type Inner: Foo = Box<Foo>;
}

trait ConstCheck<T>: Foo
where
    T: Baz,
{
    const J: i32;
}

trait Tttttttttttttttttttttttttttttttttttttttttttttttttttttttttt<T>
where
    T: Foo,
{
}

trait Ttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt<T>
where
    T: Foo,
{
}

trait FooBar<T>: Tttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt
where
    J: Bar,
{
    fn test();
}

trait WhereList<T, J>
where
    T: Foo,
    J: Bar,
{
}

trait X /* comment */ {}
trait Y // comment
{
}

// #2055
pub trait Foo:
// A and C
A + C
// and B
    + B
{}

// #2158
trait Foo {
    type ItRev = <MergingUntypedTimeSeries<SliceSeries<SliceWindow>> as UntypedTimeSeries>::IterRev;
    type IteRev =
        <MergingUntypedTimeSeries<SliceSeries<SliceWindow>> as UntypedTimeSeries>::IterRev;
}

// #2331
trait MyTrait<
    AAAAAAAAAAAAAAAAAAAA,
    BBBBBBBBBBBBBBBBBBBB,
    CCCCCCCCCCCCCCCCCCCC,
    DDDDDDDDDDDDDDDDDDDD,
>
{
    fn foo() {}
}

// Trait aliases
trait FooBar = Foo + Bar;
trait FooBar<A, B, C> = Foo + Bar;
pub trait FooBar = Foo + Bar;
pub trait FooBar<A, B, C> = Foo + Bar;
trait AAAAAAAAAAAAAAAAAA = BBBBBBBBBBBBBBBBBBB + CCCCCCCCCCCCCCCCCCCCCCCCCCCCC + DDDDDDDDDDDDDDDDDD;
pub trait AAAAAAAAAAAAAAAAAA =
    BBBBBBBBBBBBBBBBBBB + CCCCCCCCCCCCCCCCCCCCCCCCCCCCC + DDDDDDDDDDDDDDDDDD;
trait AAAAAAAAAAAAAAAAAAA =
    BBBBBBBBBBBBBBBBBBB + CCCCCCCCCCCCCCCCCCCCCCCCCCCCC + DDDDDDDDDDDDDDDDDD;
trait AAAAAAAAAAAAAAAAAA =
    BBBBBBBBBBBBBBBBBBB + CCCCCCCCCCCCCCCCCCCCCCCCCCCCC + DDDDDDDDDDDDDDDDDDD;
trait AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA<A, B, C, D, E> =
    FooBar;
trait AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA<
    A,
    B,
    C,
    D,
    E,
> = FooBar;
#[rustfmt::skip]
trait FooBar = Foo
    + Bar;

// #2637
auto trait Example {}
pub auto trait PubExample {}
pub unsafe auto trait PubUnsafeExample {}

// #3006
trait Foo<'a> {
    type Bar<'a>;
}

impl<'a> Foo<'a> for i32 {
    type Bar<'a> = i32;
}

// #3092
pub mod test {
    pub trait ATraitWithALooongName {}
    pub trait ATrait:
        ATraitWithALooongName
        + ATraitWithALooongName
        + ATraitWithALooongName
        + ATraitWithALooongName
    {
    }
}

// Trait aliases with where clauses.
trait A = where for<'b> &'b Self: Send;

trait B = where for<'b> &'b Self: Send + Clone + Copy + SomeTrait + AAAAAAAA + BBBBBBB + CCCCCCCCCC;
trait B =
    where for<'b> &'b Self: Send + Clone + Copy + SomeTrait + AAAAAAAA + BBBBBBB + CCCCCCCCCCC;
trait B = where
    for<'b> &'b Self:
        Send + Clone + Copy + SomeTrait + AAAAAAAA + BBBBBBB + CCCCCCCCCCCCCCCCCCCCCCC;
trait B = where
    for<'b> &'b Self: Send
        + Clone
        + Copy
        + SomeTrait
        + AAAAAAAA
        + BBBBBBB
        + CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC;

trait B = where
    for<'b> &'b Self: Send
        + Clone
        + Copy
        + SomeTrait
        + AAAAAAAA
        + BBBBBBB
        + CCCCCCCCC
        + DDDDDDD
        + DDDDDDDD
        + DDDDDDDDD
        + EEEEEEE;

trait A<'a, 'b, 'c> = Debug<T> + Foo where for<'b> &'b Self: Send;

trait B<'a, 'b, 'c> = Debug<T> + Foo
where
    for<'b> &'b Self: Send + Clone + Copy + SomeTrait + AAAAAAAA + BBBBBBB + CCCCCCCCC + DDDDDDD;

trait B<'a, 'b, 'c, T> = Debug<'a, T>
where
    for<'b> &'b Self: Send
        + Clone
        + Copy
        + SomeTrait
        + AAAAAAAA
        + BBBBBBB
        + CCCCCCCCC
        + DDDDDDD
        + DDDDDDDD
        + DDDDDDDDD
        + EEEEEEE;

trait Visible {
    pub const C: i32;
    pub type T;
    pub fn f();
    pub fn g() {}
}
