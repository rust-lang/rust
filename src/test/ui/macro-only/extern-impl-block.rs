// check-fail

struct T;

extern {
    impl T { //~ERROR `impl` blocks are not allowed in `extern` blocks
        fn f();
    }
}

fn main() {}
