#![feature(rustc_attrs)]
#![deny(dead_code)]

struct Data {
    value: bool, //~ ERROR: field `value` is never read
}

// This is marked as trivial
#[rustc_trivial_field_reads]
fn outer(data: &Data) -> bool {
    data.read()
}

impl Data {
    // But so is this, and it is the one that ends up using `Data::value`, so
    // the diagnostic should point here
    #[rustc_trivial_field_reads]
    fn read(&self) -> bool {
        self.value
    }
}

#[allow(unused)]
fn allowed(data: &Data) {
    let _ = data.read();
}

fn main() {
    let data = Data { value: true };
    let _ = outer(&data);
}
