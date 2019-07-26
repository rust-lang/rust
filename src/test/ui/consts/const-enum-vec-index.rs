// run-pass
#[derive(Copy, Clone)]
enum E { V1(isize), V0 }

const C: &'static [E] = &[E::V0, E::V1(0xDEADBEE)];
static C0: E = C[0];
static C1: E = C[1];
const D: &'static [E; 2] = &[E::V0, E::V1(0xDEAFBEE)];
static D0: E = D[0];
static D1: E = D[1];

pub fn main() {
    match C0 {
        E::V0 => (),
        _ => panic!()
    }
    match C1 {
        E::V1(n) => assert_eq!(n, 0xDEADBEE),
        _ => panic!()
    }

    match D0 {
        E::V0 => (),
        _ => panic!()
    }
    match D1 {
        E::V1(n) => assert_eq!(n, 0xDEAFBEE),
        _ => panic!()
    }
}
