struct GenericAssocMethod<T>(T);

impl<T> GenericAssocMethod<T> {
    fn default_hello() {}
}

fn main() {
    let x = GenericAssocMethod(33);
    x.default_hello();
    //~^ ERROR no method named `default_hello` found
}
