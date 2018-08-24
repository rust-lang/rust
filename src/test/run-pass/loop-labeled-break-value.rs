// pretty-expanded FIXME #23616

fn main() {
    'outer: loop {
        let _: i32 = loop { break 'outer };
    }
    'outer: loop {
        let _: i32 = loop { loop { break 'outer } };
    }
}
