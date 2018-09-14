struct ST1(i32, i32);

impl ST1 {
    fn ctor() -> Self {
        Self(1,2)
        //~^ ERROR: `Self` struct constructors are unstable (see issue #51994) [E0658]
    }
}

struct ST2;

impl ST2 {
    fn ctor() -> Self {
        Self
        //~^ ERROR: `Self` struct constructors are unstable (see issue #51994) [E0658]
    }
}

fn main() {
    let _ = ST1::ctor();
    let _ = ST2::ctor();
}
