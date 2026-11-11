// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

static mut a: isize = 3;

fn main() {
    // If they can't be matched against, then it's possible to capture the same
    // name as a variable, hence this should be an unreachable pattern situation
    // instead of spitting out a custom error about some identifier collisions
    // (we should allow shadowing)
    match 4 {
        a => {} //~ ERROR match bindings cannot shadow statics
        _ => {}
    }
}

struct NewBool(bool);
enum Direction {
    North,
    East,
    South,
    West
}
const NEW_FALSE: NewBool = NewBool(false);
struct Foo {
    bar: Option<Direction>,
    baz: NewBool
}

static mut STATIC_MUT_FOO: Foo = Foo { bar: Some(Direction::West), baz: NEW_FALSE };

fn mutable_statics() {
    match (Foo { bar: Some(Direction::North), baz: NewBool(true) }) {
        Foo { bar: None, baz: NewBool(true) } => (),
        STATIC_MUT_FOO => (),
        //~^ ERROR match bindings cannot shadow statics
        Foo { bar: Some(Direction::South), .. } => (),
        Foo { bar: Some(EAST), .. } => (),
        Foo { bar: Some(Direction::North), baz: NewBool(true) } => (),
        Foo { bar: Some(EAST), baz: NewBool(false) } => ()
    }
}
