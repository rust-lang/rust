// run-pass
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TS<T>(T,T);


pub fn main() {
    let ts1 = TS(1, 1);
    let ts2 = TS(1, 2);

    // in order for both PartialOrd and Ord
    let tss = [ts1, ts2];

    for (i, ts1) in tss.iter().enumerate() {
        for (j, ts2) in tss.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // PartialEq
            assert_eq!(*ts1 == *ts2, eq);
            assert_eq!(*ts1 != *ts2, !eq);

            // PartialOrd
            assert_eq!(*ts1 < *ts2, lt);
            assert_eq!(*ts1 > *ts2, gt);

            assert_eq!(*ts1 <= *ts2, le);
            assert_eq!(*ts1 >= *ts2, ge);

            // Ord
            assert_eq!(ts1.cmp(ts2), ord);
        }
    }
}
