pub enum Foo<T>
where:
//~^ ERROR unexpected colon after `where`
    T: Missing, {}
//~^ ERROR cannot find trait `Missing` in this scope
// (evidence that we continue parsing after the erroneous colon)

fn main() {}
