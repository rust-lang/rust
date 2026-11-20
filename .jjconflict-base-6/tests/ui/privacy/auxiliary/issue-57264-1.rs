mod inner {
    pub struct PubUnnameable;
}

pub struct Pub<T>(T);

impl Pub<inner::PubUnnameable> {
    pub fn pub_method() {}
}
