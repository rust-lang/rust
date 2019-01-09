// error-pattern:quux

use std::marker::PhantomData;

fn test00_start(ch: chan_t<isize>, message: isize) {
    send(ch, message);
}

type task_id = isize;
type port_id = isize;

struct chan_t<T> {
    task: task_id,
    port: port_id,
    marker: PhantomData<*mut T>,
}

fn send<T: Send>(_ch: chan_t<T>, _data: T) {
    panic!();
}

fn main() {
    panic!("quux");
}
