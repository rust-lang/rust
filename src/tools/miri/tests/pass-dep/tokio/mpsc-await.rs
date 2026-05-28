//@only-target: linux # We only support tokio on Linux
use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel(32);
    let tx2 = tx.clone();

    tokio::spawn(async move {
        tx.send("sending from handle").await.unwrap();
    });

    tokio::spawn(async move {
        tx2.send("sending from handle").await.unwrap();
    });

    while let Some(message) = rx.recv().await {
        println!("GOT = {}", message);
    }
}
