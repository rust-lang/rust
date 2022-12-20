// check-pass
#![feature(const_for)]

const fn labelled_loop() -> u32 {
    let mut n = 0;
    'outer: loop {
        'inner: loop {
            n = n + 1;
            if n > 5 && n <= 10 {
                n = n + 1;
                continue 'inner
            }
            if n > 30 {
                break 'outer
            }
        }
    }
    n
}

const X: u32 = labelled_loop();

fn main() {
    println!("{X}");
}
