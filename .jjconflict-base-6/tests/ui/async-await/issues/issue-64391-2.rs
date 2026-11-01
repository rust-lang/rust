// Regression test for #64391
//
// As described on the issue, the (spurious) `DROP` inserted for the
// `"".to_string()` value was causing a (spurious) unwind path that
// led us to believe that the future might be dropped after `config`
// had been dropped. This cannot, in fact, happen.
//
//@ check-pass
//@ edition:2018

async fn connect() {
    let config = 666;
    connect2(&config, "".to_string()).await
}

async fn connect2(_config: &u32, _tls: String) {
    unimplemented!()
}

fn main() { }
