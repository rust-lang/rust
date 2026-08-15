fn retry() -> impl Sized {}

struct Core<T>(T);

impl Core<XXX> { //~ ERROR cannot find type `XXX` in this scope
    pub fn spawn(self) {}
}

fn main() {
    let core = Core(1);
    core.spawn(retry()); //~ ERROR this method takes 0 arguments but 1 argument was supplied
}
