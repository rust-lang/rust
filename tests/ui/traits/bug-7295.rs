//@ run-pass

pub trait Foo<T> {
    fn func1<U>(&self, t: U, w: T);

    fn func2<U>(&self, t: U, w: T) {
        self.func1(t, w);
    }
}

pub fn main() {

}
