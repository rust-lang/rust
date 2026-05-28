trait Foo {
    fn a(&self) -> i32 {
        10
    }

    fn z(&self) -> i32 {
        11
    }

    fn y(&self) -> i32 {
        12
    }
}

trait Bar {
    fn b(&self) -> i32 {
        20
    }

    fn w(&self) -> i32 {
        21
    }
}

trait Baz {
    fn c(&self) -> i32 {
        30
    }
}

impl Foo for i32 {
    fn a(&self) -> i32 {
        100
    }
}

impl Bar for i32 {
    fn b(&self) -> i32 {
        200
    }
}

impl Baz for i32 {
    fn c(&self) -> i32 {
        300
    }
}

fn main() {
    let baz: &dyn Baz = &1;
    let _: &dyn std::fmt::Debug = baz;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Send = baz;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Sync = baz;
    //~^ ERROR mismatched types [E0308]

    let bar: &dyn Bar = baz;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn std::fmt::Debug = bar;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Send = bar;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Sync = bar;
    //~^ ERROR mismatched types [E0308]

    let foo: &dyn Foo = baz;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn std::fmt::Debug = foo;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Send = foo;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Sync = foo;
    //~^ ERROR mismatched types [E0308]

    let foo: &dyn Foo = bar;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn std::fmt::Debug = foo;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Send = foo;
    //~^ ERROR mismatched types [E0308]
    let _: &dyn Sync = foo;
    //~^ ERROR mismatched types [E0308]
}
