mod foo { pub struct Bar; }

fn main() {
    {
        struct Bar;
        use foo::Bar;
        //~^ ERROR the name `Bar` is defined multiple times
    }
}
