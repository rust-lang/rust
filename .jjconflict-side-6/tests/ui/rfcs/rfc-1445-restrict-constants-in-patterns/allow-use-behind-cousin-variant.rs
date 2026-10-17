// rust-lang/rust#62614: we want to allow matching on constants of types that
// have non-structural-match variants, *if* the constant itself does not use
// any such variant.

// NOTE: for now, deliberately leaving the lint `indirect_structural_match` set
// to its default, so that we will not issue a diangostic even if
// rust-lang/rust#62614 remains an open issue.

//@ run-pass

struct Sum(u32, u32);

impl PartialEq for Sum {
    fn eq(&self, other: &Self) -> bool { self.0 + self.1 == other.0 + other.1 }
}

impl Eq for Sum { }

#[derive(PartialEq, Eq)]
enum Eek {
    TheConst,
    UnusedByTheConst(Sum)
}

const THE_CONST: Eek = Eek::TheConst;
const SUM_THREE: Eek = Eek::UnusedByTheConst(Sum(3,0));

const EEK_ZERO: &[Eek] = &[];
const EEK_ONE: &[Eek] = &[THE_CONST];

pub fn main() {
    match Eek::UnusedByTheConst(Sum(1,2)) {
        ref sum if sum == &SUM_THREE => { println!("Hello 0"); }
        _ => { println!("Gbye"); }
    }

    match Eek::TheConst {
        THE_CONST => { println!("Hello 1"); }
        _ => { println!("Gbye"); }
    }


    match & &Eek::TheConst {
        & & THE_CONST => { println!("Hello 2"); }
        _ => { println!("Gbye"); }
    }

    match & & &[][..] {
        & & EEK_ZERO => { println!("Hello 3"); }
        & & EEK_ONE => { println!("Gbye"); }
        _ => { println!("Gbye"); }
    }

    match & & &[Eek::TheConst][..] {
        & & EEK_ZERO => { println!("Gby"); }
        & & EEK_ONE => { println!("Hello 4"); }
        _ => { println!("Gbye"); }
    }
}
