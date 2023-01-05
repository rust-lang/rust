// check-pass

trait SomeTrait {}

pub struct Exhibit {
    constant: usize,
    factory: fn(&usize) -> Box<dyn SomeTrait>,
}

pub const A_CONSTANT: &[Exhibit] = &[
    Exhibit {
        constant: 1,
        factory: |_| unimplemented!(),
    },
    Exhibit {
        constant: "Hello world".len(),
        factory: |_| unimplemented!(),
    },
];

fn main() {}
