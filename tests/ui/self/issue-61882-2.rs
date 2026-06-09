struct A<T>(T);

impl A<&'static u8> {
    fn f() {
        let x = 0;
        Self(&x);
        //~^ ERROR `x` does not live long enough
    }
}

fn main() {}
