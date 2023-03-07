mod m {
    pub union U {
        pub a: u8,
        pub(super) b: u8,
        c: u8,
    }
}

fn main() {
    let u = m::U { a: 10 };

    let a = u.a; // OK
    let b = u.b; // OK
    let c = u.c; //~ ERROR field `c` of union `U` is private
}
