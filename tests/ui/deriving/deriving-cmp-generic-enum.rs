//@ run-pass
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum E<T> {
    E0,
    E1(T),
    E2(T,T)
}

pub fn main() {
    let e0 = E::E0;
    let e11 = E::E1(1);
    let e12 = E::E1(2);
    let e21 = E::E2(1, 1);
    let e22 = E::E2(1, 2);

    // in order for both PartialOrd and Ord
    let es = [e0, e11, e12, e21, e22];

    for (i, e1) in es.iter().enumerate() {
        for (j, e2) in es.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // PartialEq
            assert_eq!(*e1 == *e2, eq);
            assert_eq!(*e1 != *e2, !eq);

            // PartialOrd
            assert_eq!(*e1 < *e2, lt);
            assert_eq!(*e1 > *e2, gt);

            assert_eq!(*e1 <= *e2, le);
            assert_eq!(*e1 >= *e2, ge);

            // Ord
            assert_eq!(e1.cmp(e2), ord);
        }
    }
}
