macro_rules! define {
    (struct $name:ident) => {
        struct $name {}
    };
}

define! {
    struct Hello
}

fn main() {
    let _hello_instance = Hello {};
}
