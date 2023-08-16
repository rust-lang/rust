// run-fail
//@error-in-other-file:thread 'owned name' panicked at 'test'
//@ignore-target-emscripten Needs threads.

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
