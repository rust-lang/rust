enum Foo<'a> {
    Bar { field: &'a u32 }
}

fn in_let() {
   let y = 22;
   let foo = Foo::Bar { field: &y };
   //~^ ERROR `y` does not live long enough
   let Foo::Bar::<'static> { field: _z } = foo;
}

fn in_match() {
   let y = 22;
   let foo = Foo::Bar { field: &y };
   //~^ ERROR `y` does not live long enough
   match foo {
       Foo::Bar::<'static> { field: _z } => {}
   }
}

fn in_let_2() {
   let y = 22;
   let foo = Foo::Bar { field: &y };
   //~^ ERROR `y` does not live long enough
   let Foo::<'static>::Bar { field: _z } = foo;
}

fn in_match_2() {
   let y = 22;
   let foo = Foo::Bar { field: &y };
   //~^ ERROR `y` does not live long enough
   match foo {
       Foo::<'static>::Bar { field: _z } => {}
   }
}

fn in_let_3() {
    let y = 22;
    let foo = Foo::Bar { field: &y };
    let Foo::Bar { field: _z } = foo;
}

fn in_match_3() {
    let y = 22;
    let foo = Foo::Bar { field: &y };
    match foo {
        Foo::Bar { field: _z } => {}
    }
}

fn main() { }
