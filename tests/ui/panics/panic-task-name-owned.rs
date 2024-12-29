//@ run-fail
//@ check-run-results:thread 'owned name' panicked
//@ check-run-results:test
//@ needs-threads

use std::thread::Builder;

fn main() {
    let r: () = Builder::new()
                    .name("owned name".to_string())
                    .spawn(move || {
                        panic!("test");
                        ()
                    })
                    .unwrap()
                    .join()
                    .unwrap();
    panic!();
}
