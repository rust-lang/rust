enum foo = int;

fn main() {
    let x = foo(3);
    *x = 4; //! ERROR assigning to enum content
}