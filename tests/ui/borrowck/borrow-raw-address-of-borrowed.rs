fn address_of_shared() {
    let mut x = 0;
    let y = &x;

    let q = &raw mut x;                 //~ ERROR cannot borrow

    drop(y);
}

fn address_of_mutably_borrowed() {
    let mut x = 0;
    let y = &mut x;

    let p = &raw const x;               //~ ERROR cannot borrow
    let q = &raw mut x;                 //~ ERROR cannot borrow

    drop(y);
}

fn main() {}
