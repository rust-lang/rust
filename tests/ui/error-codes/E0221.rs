trait T1 {}
trait T2 {}

trait Foo {
    type A: T1;
}

trait Bar : Foo {
    type A: T2;
    fn do_something() {
        let _: Self::A;
        //~^ ERROR E0221
    }
}

trait T3 {}

trait My : std::str::FromStr {
    type Err: T3;
    fn test() {
        let _: Self::Err;
        //~^ ERROR E0221
    }
}

fn main() {
}
