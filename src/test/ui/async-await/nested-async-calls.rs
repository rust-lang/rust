// edition:2018

async fn first() {
    second().await;
}

async fn second() {
    third().await;
}

async fn third() {
    struct NotSend(*const ());
    struct Outer(NotSend);
    async fn dummy() {}

    let _a: Outer;
    dummy().await;
}

fn require_send<T: Send>(_val: T) {}

fn main() {
    struct Wrapper<T>(T);
    let wrapped = Wrapper(first());

    require_send(wrapped);
    //~^ ERROR future cannot be sent between threads safely
}
