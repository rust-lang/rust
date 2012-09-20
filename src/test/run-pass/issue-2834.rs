// Test case for issue #2843.
//

proto! streamp (
    open:send<T: Send> {
        data(T) -> open<T>
    }
)

fn rendezvous() {
    let (c, s) = streamp::init();
    let streams: ~[streamp::client::open<int>] = ~[c];

    error!("%?", streams[0]);
}

fn main() {
    //os::getenv("FOO");
    rendezvous();
}
