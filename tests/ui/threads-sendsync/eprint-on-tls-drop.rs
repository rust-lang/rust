// run-pass
// ignore-emscripten no processes
// ignore-sgx no processes

use std::cell::RefCell;
use std::env;
use std::process::Command;

fn main() {
    let name = "YOU_ARE_THE_TEST";
    if env::var(name).is_ok() {
        std::thread::spawn(|| {
            TLS.with(|f| f.borrow().ensure());
        })
        .join()
        .unwrap();
    } else {
        let me = env::current_exe().unwrap();
        let output = Command::new(&me).env(name, "1").output().unwrap();
        println!("{:?}", output);
        assert!(output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("hello new\n"));
        assert!(stderr.contains("hello drop\n"));
    }
}

struct Stuff {
    _x: usize,
}

impl Stuff {
    fn new() -> Self {
        eprintln!("hello new");
        Self { _x: 0 }
    }

    fn ensure(&self) {}
}

impl Drop for Stuff {
    fn drop(&mut self) {
        eprintln!("hello drop");
    }
}

thread_local! {
    static TLS: RefCell<Stuff> = RefCell::new(Stuff::new());
}
