// issue-52240: Can turn immutable into mut with `ref mut`

enum Foo {
    Bar(i32),
}

fn main() {
    let arr = vec!(Foo::Bar(0));
    if let (Some(Foo::Bar(ref mut val)), _) = (&arr.get(0), 0) {
        //~^ ERROR cannot borrow data in a `&` reference as mutable
        *val = 9001;
    }
    match arr[0] {
        Foo::Bar(ref s) => println!("{}", s)
    }
}
