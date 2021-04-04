// check-pass

#![feature(const_refs_to_cell)]

const FOO: () = {
    let x = std::cell::Cell::new(42);
    let y = &x;
};

fn main() {
    FOO;
}
