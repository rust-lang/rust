// This used to mis-compile because the mir-opt `SimplifyArmIdentity`
// did not check that the types matched up in the `Ok(r)` branch.
//
//@ run-pass
//@ compile-flags: -Zmir-opt-level=3

#[derive(Debug, PartialEq, Eq)]
enum SpecialsRes { Res(u64) }

fn e103() -> SpecialsRes {
    if let Ok(r) = "1".parse() {
        SpecialsRes::Res(r)
    } else {
        SpecialsRes::Res(42)
    }
}

fn main() {
    assert_eq!(e103(), SpecialsRes::Res(1));
}
