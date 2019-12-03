// edition:2018

struct Ia<S>(u32, S);

impl<S> Ia<S> {
    fn partial(_: S) {}
    fn full(self) {}

    async fn crash(self) {
        Self::partial(self.1);
        Self::full(self); //~ ERROR use of moved value: `self`
    }
}

fn main() {}
