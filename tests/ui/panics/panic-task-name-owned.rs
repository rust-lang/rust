//@ run-fail
//@ check-run-results
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
