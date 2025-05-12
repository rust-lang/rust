use bar::Foo; //~ ERROR unresolved import `bar::Foo` [E0432]
              //~^ NOTE no `Foo` in `bar`
mod bar {
    use Foo;
}

fn main() {}
