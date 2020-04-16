// run-fail
// error-pattern:quux

use std::marker::PhantomData;

fn test00_start(ch: Chan<isize>, message: isize) {
    send(ch, message);
}

type TaskId = isize;
type PortId = isize;

struct Chan<T> {
    task: TaskId,
    port: PortId,
    marker: PhantomData<*mut T>,
}

fn send<T: Send>(_ch: Chan<T>, _data: T) {
    panic!();
}

fn main() {
    panic!("quux");
}
