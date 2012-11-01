/// A thread pool abstraction. Useful for achieving predictable CPU
/// parallelism.

use pipes::{Chan, Port};
use task::{SchedMode, SingleThreaded};

enum Msg<T> {
    Execute(~fn(&T)),
    Quit
}

pub struct ThreadPool<T> {
    channels: ~[Chan<Msg<T>>],
    mut next_index: uint,

    drop {
        for self.channels.each |channel| {
            channel.send(Quit);
        }
    }
}

pub impl<T> ThreadPool<T> {
    /// Spawns a new thread pool with `n_tasks` tasks. If the `sched_mode`
    /// is None, the tasks run on this scheduler; otherwise, they run on a
    /// new scheduler with the given mode. The provided `init_fn_factory`
    /// returns a function which, given the index of the task, should return
    /// local data to be kept around in that task.
    static fn new(n_tasks: uint,
                  opt_sched_mode: Option<SchedMode>,
                  init_fn_factory: ~fn() -> ~fn(uint) -> T) -> ThreadPool<T> {
        assert n_tasks >= 1;

        let channels = do vec::from_fn(n_tasks) |i| {
            let (chan, port) = pipes::stream::<Msg<T>>();
            let init_fn = init_fn_factory();

            let task_body: ~fn() = |move port, move init_fn| {
                let local_data = init_fn(i);
                loop {
                    match port.recv() {
                        Execute(move f) => f(&local_data),
                        Quit => break
                    }
                }
            };

            // Start the task.
            match opt_sched_mode {
                None => {
                    // Run on this scheduler.
                    task::spawn(move task_body);
                }
                Some(sched_mode) => {
                    task::task().sched_mode(sched_mode).spawn(move task_body);
                }
            }

            move chan
        };

        return ThreadPool { channels: move channels, next_index: 0 };
    }

    /// Executes the function `f` on a thread in the pool. The function
    /// receives a reference to the local data returned by the `init_fn`.
    fn execute(&self, f: ~fn(&T)) {
        self.channels[self.next_index].send(Execute(move f));
        self.next_index += 1;
        if self.next_index == self.channels.len() { self.next_index = 0; }
    }
}

#[test]
fn test_thread_pool() {
    let f: ~fn() -> ~fn(uint) -> uint = || {
        let g: ~fn(uint) -> uint = |i| i;
        move g
    };
    let pool = ThreadPool::new(4, Some(SingleThreaded), move f);
    for 8.times {
        pool.execute(|i| io::println(fmt!("Hello from thread %u!", *i)));
    }
}

