// Regression test for #62312

//@ check-pass
//@ edition:2018

async fn make_boxed_object() -> Box<dyn Send> {
    Box::new(()) as _
}

async fn await_object() {
    let _ = make_boxed_object().await;
}

fn main() {}
