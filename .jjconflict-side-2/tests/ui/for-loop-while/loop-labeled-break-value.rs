//@ run-pass

fn main() {
    'outer: loop {
        let _: i32 = loop { break 'outer };
    }
    'outer2: loop {
        let _: i32 = loop { loop { break 'outer2 } };
    }
}
