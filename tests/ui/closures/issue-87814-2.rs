//@ check-pass

fn main() {
    let mut schema_all: (Vec<String>, Vec<String>) = (vec![], vec![]);

    let _c = || match schema_all.0.try_reserve(1) as Result<(), _> {
        Ok(()) => (),
        Err(_) => (),
    };
}
