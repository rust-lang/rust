//@ check-pass

#![warn(unused_parens)]
#![warn(break_with_label_and_loop)]

fn xyz() -> usize {
    'foo: {
        // parens are necessary
        break 'foo ({
            println!("Hello!");
            123
        });
    }
}

fn main() {
    xyz();
}
