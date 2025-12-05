//@ check-pass

pub fn todo<T>() -> T {
    const { todo!() }
}

fn main() {
    let _: usize = const { 0 };
}
