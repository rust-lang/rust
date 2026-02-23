type const FOO: u8 = 10;
//~^ ERROR `type const` syntax is experimental [E0658]
//~| ERROR top-level `type const` are unstable [E0658]

trait Bar {
    type const BAR: bool;
    //~^ ERROR `type const` syntax is experimental [E0658]
    //~| ERROR associated `type const` are unstable [E0658]
}

impl Bar for bool {
    type const BAR: bool = false;
    //~^ ERROR `type const` syntax is experimental [E0658]
    //~| ERROR associated `type const` are unstable [E0658]
}

fn main() { }
