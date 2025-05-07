enum Foo { Bar }

fn main() {
    Foo::Bar.a;
    //~^ ERROR no field `a` on type `Foo`
}
