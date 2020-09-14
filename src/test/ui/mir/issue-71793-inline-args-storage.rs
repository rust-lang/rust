// Verifies that inliner emits StorageLive & StorageDead when introducing
// temporaries for arguments, so that they don't become part of the generator.
// Regression test for #71793.
//
// check-pass
// edition:2018
// compile-args: -Zmir-opt-level=2

#![crate_type = "lib"]

pub async fn connect() {}

pub async fn connect_many() {
    Vec::<String>::new().first().ok_or("").unwrap();
    connect().await;
}
