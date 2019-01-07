// run-rustfix

#![warn(clippy::useless_vec)]

#[derive(Debug)]
struct NonCopy;

fn on_slice(_: &[u8]) {}
#[allow(clippy::ptr_arg)]
fn on_vec(_: &Vec<u8>) {}

struct Line {
    length: usize,
}

impl Line {
    fn length(&self) -> usize {
        self.length
    }
}

fn main() {
    on_slice(&vec![]);
    on_slice(&[]);

    on_slice(&vec![1, 2]);
    on_slice(&[1, 2]);

    on_slice(&vec![1, 2]);
    on_slice(&[1, 2]);
    #[rustfmt::skip]
    on_slice(&vec!(1, 2));
    on_slice(&[1, 2]);

    on_slice(&vec![1; 2]);
    on_slice(&[1; 2]);

    on_vec(&vec![]);
    on_vec(&vec![1, 2]);
    on_vec(&vec![1; 2]);

    // Now with non-constant expressions
    let line = Line { length: 2 };

    on_slice(&vec![2; line.length]);
    on_slice(&vec![2; line.length()]);

    for a in vec![1, 2, 3] {
        println!("{:?}", a);
    }

    for a in vec![NonCopy, NonCopy] {
        println!("{:?}", a);
    }
}
