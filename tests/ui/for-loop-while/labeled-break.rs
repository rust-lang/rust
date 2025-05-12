//@ run-pass

pub fn main() {
    'foo: loop {
        loop {
            break 'foo;
        }
    }

    'bar: for _ in 0..100 {
        loop {
            break 'bar;
        }
    }

    'foobar: while 1 + 1 == 2 {
        loop {
            break 'foobar;
        }
    }
}
