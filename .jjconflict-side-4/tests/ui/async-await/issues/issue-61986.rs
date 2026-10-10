//@ build-pass (FIXME(62277): could be check-pass?)
//@ edition:2018
//
// Tests that we properly handle StorageDead/StorageLives for temporaries
// created in async loop bodies.

async fn bar() -> Option<()> {
    Some(())
}

async fn listen() {
    while let Some(_) = bar().await {
        String::new();
    }
}

fn main() {
    listen();
}
