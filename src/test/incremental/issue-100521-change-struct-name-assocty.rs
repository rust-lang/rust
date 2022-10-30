// revisions: rpass1 rpass2

pub fn foo() {
    bar();
    baz::<()>();
}

fn bar()
where
    <() as Table>::AllColumns:,
{
}

fn baz<W>()
where
    W: AsQuery,
    <W as AsQuery>::Query:,
{
}

trait AsQuery {
    type Query;
}

trait UnimplementedTrait {}

impl<T> AsQuery for T
where
    T: UnimplementedTrait,
{
    type Query = ();
}

struct Wrapper<Expr>(Expr);

impl<Ret> AsQuery for Wrapper<Ret> {
    type Query = ();
}

impl AsQuery for ()
where
    Wrapper<<() as Table>::AllColumns>: AsQuery,
{
    type Query = ();
}

trait Table {
    type AllColumns;
}

#[cfg(rpass1)]
impl Table for () {
    type AllColumns = Checksum1;
}
#[cfg(rpass1)]
struct Checksum1;

#[cfg(rpass2)]
impl Table for () {
    type AllColumns = Checksum2;
}
#[cfg(rpass2)]
struct Checksum2;

fn main() {}
