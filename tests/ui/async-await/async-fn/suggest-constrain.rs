// Ensure that we don't suggest constraining `CallRefFuture` here,
// since that isn't stable.

fn spawn<F: AsyncFn() + Send>(f: F) {
    check_send(f());
    //~^ ERROR cannot be sent between threads safely
}

fn check_send<T: Send>(_: T) {}

fn main() {}
