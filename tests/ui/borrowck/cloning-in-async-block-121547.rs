//@ edition:2021

async fn clone_async_block(value: String) {
    for _ in 0..10 {
        async { //~ ERROR: use of moved value: `value` [E0382]
            drop(value);
            //~^ HELP: consider cloning the value if the performance cost is acceptable
        }.await
    }
}
fn main() {}
