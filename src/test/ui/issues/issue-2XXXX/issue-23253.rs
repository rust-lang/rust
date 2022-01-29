enum Foo { Bar }

fn main() {
    Foo::Bar.a;
    //~^ no field `a` on type `Foo`
}
