struct TestClient;

impl TestClient {
    fn get_inner_ref(&self) -> &Vec<usize> {
        todo!()
    }
}

fn main() {
    let client = TestClient;
    let inner = client.get_inner_ref();
    //~^ HELP consider changing this to be a mutable reference
    inner.clear();
    //~^ ERROR cannot borrow `*inner` as mutable, as it is behind a `&` reference [E0596]
}
