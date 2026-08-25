//@ run-pass
//@ check-run-results
//@ only-windows

struct Bye;

impl Drop for Bye {
    fn drop(&mut self) {
        print!(", world!");
    }
}

fn main() {
    thread_local!{
        static BYE: Bye = Bye;
    }
    BYE.with(|_| {
        print!("hello");
    });
}
