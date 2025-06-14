#[derive(Default)]
struct Foo{
    x: String,
}

impl Foo {
    fn foo(&mut self, name: String) {
        self.x = name;
    }
}



fn main() {
    let name1 = String::from("foo");
    for _ in 0..3 {
        let mut foo = Foo::default();
        foo.foo(name1); //~ ERROR use of moved value: `name1` [E0382]
        println!("{}", foo.x);
    }

    let name2 = String::from("bar");
    for mut foo in [1,2,3].iter_mut().map(|m|  Foo::default()) {
        foo.foo(name2); //~ ERROR use of moved value: `name2` [E0382]
        println!("{}", foo.x);
    }


    let name3 = String::from("baz");
    let mut foo = Foo::default();
    for _ in 0..10 {
        foo.foo(name3); //~ ERROR use of moved value: `name3` [E0382]
        println!("{}", foo.x);
    }
}
