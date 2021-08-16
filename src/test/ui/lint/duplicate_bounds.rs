#![deny(duplicate_bounds)]

trait DupDirectAndWhere {}
fn dup_direct_and_where<T: DupDirectAndWhere>(t: T)
where
    T: DupDirectAndWhere,
    //~^ ERROR this trait bound has already been specified
    T: DupDirectAndWhere,
    //~^ ERROR this trait bound has already been specified
{
    unimplemented!();
}

trait DupDirect {}
fn dup_direct<T: DupDirect + DupDirect>(t: T) {
    //~^ ERROR this trait bound has already been specified
    unimplemented!();
}

trait DupWhere {}
fn dup_where<T>(t: T)
where
    T: DupWhere + DupWhere,
    //~^ ERROR this trait bound has already been specified
{
    unimplemented!();
}

trait NotDup {}
fn not_dup<T: NotDup, U: NotDup>((t, u): (T, U)) {
    unimplemented!();
}

fn dup_lifetimes<'a, 'b: 'a + 'a>()
//~^ ERROR this lifetime bound has already been specified
where
    'b: 'a,
    //~^ ERROR this lifetime bound has already been specified
{
}

fn dup_lifetimes_generic<'a, T: 'a + 'a>()
//~^ ERROR this lifetime bound has already been specified
where
    T: 'a,
    //~^ ERROR this lifetime bound has already been specified
{
}

trait Everything {}
fn everything<T: Everything + Everything, U: Everything + Everything>((t, u): (T, U))
//~^ ERROR this trait bound has already been specified
//~| ERROR this trait bound has already been specified
where
    T: Everything + Everything + Everything,
    //~^ ERROR this trait bound has already been specified
    //~| ERROR this trait bound has already been specified
    //~| ERROR this trait bound has already been specified
    U: Everything,
    //~^ ERROR this trait bound has already been specified
{
    unimplemented!();
}

trait DupStructBound {}
struct DupStruct<T: DupStructBound + DupStructBound>(T)
//~^ ERROR this trait bound has already been specified
where
    T: DupStructBound;
//~^ ERROR this trait bound has already been specified

impl<'a, T: 'a + DupStructBound + DupStructBound> DupStruct<T>
//~^ ERROR this trait bound has already been specified
where
    T: 'a + DupStructBound,
    //~^ ERROR this lifetime bound has already been specified
    //~| ERROR this trait bound has already been specified
{
    fn _x() {}
}

fn main() {}
