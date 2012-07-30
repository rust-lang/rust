// Test case for issue #2843.
//
// xfail-test

proto! streamp {
    open:send<T: send> {
        data(T) -> open<T>
    }
}

fn rendezvous() {
    let (c, s) = streamp::init();
    let streams: ~[streamp::client::open<int>] = ~[c];

    error!{"%?", streams[0]};
}

fn main(args: ~[str]) {
    //os::getenv("FOO");
    rendezvous();
}