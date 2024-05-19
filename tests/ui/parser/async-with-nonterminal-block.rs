//@ check-pass
//@ edition:2021

macro_rules! create_async {
    ($body:block) => {
        async $body
    };
}

async fn other() {}

fn main() {
    let y = create_async! {{
        other().await;
    }};
}
