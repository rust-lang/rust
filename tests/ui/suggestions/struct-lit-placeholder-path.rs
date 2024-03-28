// Regression test for issue #98282.

mod blah {
    pub struct Stuff { x: i32 }
    pub fn do_stuff(_: Stuff) {}
}

fn main() {
    blah::do_stuff(_ { x: 10 });
    //~^ ERROR the placeholder `_` is not allowed for the path in struct literals
    //~| NOTE not allowed in struct literals
    //~| HELP replace it with the correct type
}

#[cfg(FALSE)]
fn disabled() {
    blah::do_stuff(_ { x: 10 });
    //~^ ERROR the placeholder `_` is not allowed for the path in struct literals
    //~| NOTE not allowed in struct literals
    //~| HELP replace it with an appropriate type
}
