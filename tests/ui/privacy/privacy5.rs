//@ aux-build:privacy_tuple_struct.rs

extern crate privacy_tuple_struct as other;

mod a {
    pub struct A(());
    pub struct B(isize);
    pub struct C(pub isize, isize);
    pub struct D(pub isize);

    fn test() {
        let a = A(());
        let b = B(2);
        let c = C(2, 3);
        let d = D(4);

        let A(()) = a;
        let A(_) = a;
        match a { A(()) => {} }
        match a { A(_) => {} }

        let B(_) = b;
        let B(_b) = b;
        match b { B(_) => {} }
        match b { B(_b) => {} }
        match b { B(1) => {} B(_) => {} }

        let C(_, _) = c;
        let C(_a, _) = c;
        let C(_, _b) = c;
        let C(_a, _b) = c;
        match c { C(_, _) => {} }
        match c { C(_a, _) => {} }
        match c { C(_, _b) => {} }
        match c { C(_a, _b) => {} }

        let D(_) = d;
        let D(_d) = d;
        match d { D(_) => {} }
        match d { D(_d) => {} }
        match d { D(1) => {} D(_) => {} }

        let a2 = A;
        let b2 = B;
        let c2 = C;
        let d2 = D;
    }
}

fn this_crate() {
    let a = a::A(()); //~ ERROR tuple struct constructor `A` is private
    let b = a::B(2); //~ ERROR tuple struct constructor `B` is private
    let c = a::C(2, 3); //~ ERROR tuple struct constructor `C` is private
    let d = a::D(4);

    let a::A(()) = a; //~ ERROR tuple struct constructor `A` is private
    let a::A(_) = a; //~ ERROR tuple struct constructor `A` is private
    match a { a::A(()) => {} } //~ ERROR tuple struct constructor `A` is private
    match a { a::A(_) => {} } //~ ERROR tuple struct constructor `A` is private

    let a::B(_) = b; //~ ERROR tuple struct constructor `B` is private
    let a::B(_b) = b; //~ ERROR tuple struct constructor `B` is private
    match b { a::B(_) => {} } //~ ERROR tuple struct constructor `B` is private
    match b { a::B(_b) => {} } //~ ERROR tuple struct constructor `B` is private
    match b { a::B(1) => {} a::B(_) => {} } //~ ERROR tuple struct constructor `B` is private
                                            //~^ ERROR tuple struct constructor `B` is private

    let a::C(_, _) = c; //~ ERROR tuple struct constructor `C` is private
    let a::C(_a, _) = c; //~ ERROR tuple struct constructor `C` is private
    let a::C(_, _b) = c; //~ ERROR tuple struct constructor `C` is private
    let a::C(_a, _b) = c; //~ ERROR tuple struct constructor `C` is private
    match c { a::C(_, _) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { a::C(_a, _) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { a::C(_, _b) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { a::C(_a, _b) => {} } //~ ERROR tuple struct constructor `C` is private

    let a::D(_) = d;
    let a::D(_d) = d;
    match d { a::D(_) => {} }
    match d { a::D(_d) => {} }
    match d { a::D(1) => {} a::D(_) => {} }

    let a2 = a::A; //~ ERROR tuple struct constructor `A` is private
    let b2 = a::B; //~ ERROR tuple struct constructor `B` is private
    let c2 = a::C; //~ ERROR tuple struct constructor `C` is private
    let d2 = a::D;
}

fn xcrate() {
    let a = other::A(()); //~ ERROR tuple struct constructor `A` is private
    let b = other::B(2); //~ ERROR tuple struct constructor `B` is private
    let c = other::C(2, 3); //~ ERROR tuple struct constructor `C` is private
    let d = other::D(4);

    let other::A(()) = a; //~ ERROR tuple struct constructor `A` is private
    let other::A(_) = a; //~ ERROR tuple struct constructor `A` is private
    match a { other::A(()) => {} } //~ ERROR tuple struct constructor `A` is private
    match a { other::A(_) => {} } //~ ERROR tuple struct constructor `A` is private

    let other::B(_) = b; //~ ERROR tuple struct constructor `B` is private
    let other::B(_b) = b; //~ ERROR tuple struct constructor `B` is private
    match b { other::B(_) => {} } //~ ERROR tuple struct constructor `B` is private
    match b { other::B(_b) => {} } //~ ERROR tuple struct constructor `B` is private
    match b { other::B(1) => {}//~ ERROR tuple struct constructor `B` is private
        other::B(_) => {} }    //~ ERROR tuple struct constructor `B` is private

    let other::C(_, _) = c; //~ ERROR tuple struct constructor `C` is private
    let other::C(_a, _) = c; //~ ERROR tuple struct constructor `C` is private
    let other::C(_, _b) = c; //~ ERROR tuple struct constructor `C` is private
    let other::C(_a, _b) = c; //~ ERROR tuple struct constructor `C` is private
    match c { other::C(_, _) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { other::C(_a, _) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { other::C(_, _b) => {} } //~ ERROR tuple struct constructor `C` is private
    match c { other::C(_a, _b) => {} } //~ ERROR tuple struct constructor `C` is private

    let other::D(_) = d;
    let other::D(_d) = d;
    match d { other::D(_) => {} }
    match d { other::D(_d) => {} }
    match d { other::D(1) => {} other::D(_) => {} }

    let a2 = other::A; //~ ERROR tuple struct constructor `A` is private
    let b2 = other::B; //~ ERROR tuple struct constructor `B` is private
    let c2 = other::C; //~ ERROR tuple struct constructor `C` is private
    let d2 = other::D;
}

fn main() {}
