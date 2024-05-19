struct TestClient;

impl TestClient {
    fn get_inner_ref(&self) -> &Vec<usize> {
        todo!()
    }
}

fn main() {
    let client = TestClient;
    let inner = client.get_inner_ref();
    //~^ HELP consider specifying this binding's type
    inner.clear();
    //~^ ERROR cannot borrow `*inner` as mutable, as it is behind a `&` reference [E0596]
    //~| NOTE `inner` is a `&` reference, so the data it refers to cannot be borrowed as mutable
}
