// Tests for #88015 when using if let chains in match guards

//@run-pass

#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

fn lhs_let(opt: Option<bool>) {
    match opt {
        None | Some(false) | Some(true) if let x = 42 && true => assert_eq!(x, 42),
        _ => panic!()
    }
}

fn rhs_let(opt: Option<bool>) {
    match opt {
        None | Some(false) | Some(true) if true && let x = 41 => assert_eq!(x, 41),
        _ => panic!()
    }
}

fn main() {
    lhs_let(None);
    lhs_let(Some(false));
    lhs_let(Some(true));
    rhs_let(None);
    rhs_let(Some(false));
    rhs_let(Some(true));
}
