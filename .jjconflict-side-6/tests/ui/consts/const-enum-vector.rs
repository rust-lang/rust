//@ run-pass

enum E { V1(isize), V0 }
static C: [E; 3] = [E::V0, E::V1(0xDEADBEE), E::V0];

pub fn main() {
    match C[1] {
        E::V1(n) => assert_eq!(n, 0xDEADBEE),
        _ => panic!()
    }
    match C[2] {
        E::V0 => (),
        _ => panic!()
    }
}
