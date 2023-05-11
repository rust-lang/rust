fn main() {
    insert_resource(Marker);
    insert_resource(Time);
    //~^ ERROR the trait bound `fn(u32) -> Time {Time}: Resource` is not satisfied
    //~| HELP use parentheses to construct this tuple struct
}

trait Resource {}

fn insert_resource<R: Resource>(resource: R) {}

struct Marker;
impl Resource for Marker {}

struct Time(u32);

impl Resource for Time {}
