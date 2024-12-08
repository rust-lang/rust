//@ known-bug: rust-lang/rust#128094
//@ compile-flags: -Zmir-opt-level=5 --edition=2018

pub enum Request {
    TestSome(T),
}

pub async fn handle_event(event: Request) {
    async move {
        static instance: Request = Request { bar: 17 };
        &instance
    }
    .await;
}
