//@ run-fail
//@ error-pattern:oh, dear
//@ needs-subprocess

fn main() -> ! {
    panic!("oh, dear");
}
