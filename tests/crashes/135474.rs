//@ known-bug: #135474
fn retry() -> impl Sized {}

struct Core<T>(T);

// Invalid type argument
impl Core<XXX> {
    pub fn spawn(self) {}
}

fn main() {
    let core = Core(1);
    // extraneous argument
    core.spawn(retry());
}
