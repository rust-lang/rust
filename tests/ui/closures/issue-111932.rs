trait Foo: std::fmt::Debug {}

fn print_foos(foos: impl Iterator<Item = dyn Foo>) {
    foos.for_each(|foo| { //~ ERROR [E0277]
        println!("{:?}", foo); //~ ERROR [E0277]
    });
}

fn main() {}
