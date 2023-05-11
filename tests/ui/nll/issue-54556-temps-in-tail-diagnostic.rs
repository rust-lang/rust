fn main() {
    {
        let mut _thing1 = D(Box::new("thing1"));
        // D("other").next(&_thing1).end()
        D(&_thing1).end() //~ ERROR does not live long enough
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
