//@ check-pass
//@ edition:2018

struct Test(String);

impl Test {
    async fn borrow_async(&self) {}

    fn with(&mut self, s: &str) -> &mut Self {
        self.0 = s.into();
        self
    }
}

async fn test() {
    Test("".to_string()).with("123").borrow_async().await;
}

fn main() { }
