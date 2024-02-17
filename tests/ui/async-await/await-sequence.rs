//@ edition:2021
//@ build-pass

use std::collections::HashMap;

fn main() {
    let _ = real_main();
}

async fn nop() {}

async fn real_main() {
    nop().await;
    nop().await;
    nop().await;
    nop().await;

    let mut map: HashMap<(), ()> = HashMap::new();
    map.insert((), nop().await);
}
