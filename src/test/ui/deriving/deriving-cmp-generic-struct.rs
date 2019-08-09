// run-pass
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct S<T> {
    x: T,
    y: T
}

pub fn main() {
    let s1 = S {x: 1, y: 1};
    let s2 = S {x: 1, y: 2};

    // in order for both PartialOrd and Ord
    let ss = [s1, s2];

    for (i, s1) in ss.iter().enumerate() {
        for (j, s2) in ss.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // PartialEq
            assert_eq!(*s1 == *s2, eq);
            assert_eq!(*s1 != *s2, !eq);

            // PartialOrd
            assert_eq!(*s1 < *s2, lt);
            assert_eq!(*s1 > *s2, gt);

            assert_eq!(*s1 <= *s2, le);
            assert_eq!(*s1 >= *s2, ge);

            // Ord
            assert_eq!(s1.cmp(s2), ord);
        }
    }
}
