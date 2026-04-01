use std::thread;
use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    workers: Vec<thread::JoinHandle<()>>,
    queue: Arc<()>,
}

impl ThreadPool {
    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        panic!()
    }
}

fn main() {
    let results = Arc::new(Mutex::new(Vec::new())); //~ NOTE move occurs because
    let pool = ThreadPool {
        workers: vec![],
        queue: Arc::new(()),
    };

    for i in 0..20 { //~ NOTE inside of this loop
        // let results = Arc::clone(&results); // Forgot this.
        pool.execute(move || { //~ ERROR E0382
            //~^ NOTE value moved into closure here, in previous iteration of loop
            //~| HELP consider cloning the value before moving it into the closure
            let mut r = results.lock().unwrap(); //~ NOTE use occurs due to use in closure
            r.push(i);
        });
    }
}
