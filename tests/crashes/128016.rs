//@ known-bug: #128016
macro_rules! len {
    () => {
        target
    };
}

fn main() {
    let val: [str; len!()] = [];
}
