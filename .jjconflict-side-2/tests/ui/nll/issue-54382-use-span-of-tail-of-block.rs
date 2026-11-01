fn main() {
    {
        let mut _thing1 = D(Box::new("thing1"));
        {
            let _thing2 = D("thing2");
            side_effects();
            D("other").next(&_thing1)
//~^ ERROR does not live long enough
        }
    }

    ;
}

#[derive(Debug)]
struct D<T: std::fmt::Debug>(T);

impl<T: std::fmt::Debug>  Drop for D<T> {
    fn drop(&mut self) {
        println!("dropping {:?})", self);
    }
}

impl<T: std::fmt::Debug> D<T> {
    fn next<U: std::fmt::Debug>(&self, _other: U) -> D<U> { D(_other) }
    fn end(&self) { }
}

fn side_effects() { }
