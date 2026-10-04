//@ run-pass
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum ES<T> {
    ES1 { x: T },
    ES2 { x: T, y: T }
}


pub fn main() {
    let (es11, es12, es21, es22) = (ES::ES1 {
        x: 1
    }, ES::ES1 {
        x: 2
    }, ES::ES2 {
        x: 1,
        y: 1
    }, ES::ES2 {
        x: 1,
        y: 2
    });

    // in order for both PartialOrd and Ord
    let ess = [es11, es12, es21, es22];

    for (i, es1) in ess.iter().enumerate() {
        for (j, es2) in ess.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let (lt, le) = (i < j, i <= j);
            let (gt, ge) = (i > j, i >= j);

            // PartialEq
            assert_eq!(*es1 == *es2, eq);
            assert_eq!(*es1 != *es2, !eq);

            // PartialOrd
            assert_eq!(*es1 < *es2, lt);
            assert_eq!(*es1 > *es2, gt);

            assert_eq!(*es1 <= *es2, le);
            assert_eq!(*es1 >= *es2, ge);

            // Ord
            assert_eq!(es1.cmp(es2), ord);
        }
    }
}
