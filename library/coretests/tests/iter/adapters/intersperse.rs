use core::iter::*;

#[test]
fn test_intersperse() {
    let v = std::iter::empty().intersperse(0u32).collect::<Vec<_>>();
    assert_eq!(v, vec![]);

    let v = std::iter::once(1).intersperse(0).collect::<Vec<_>>();
    assert_eq!(v, vec![1]);

    let xs = ["a", "", "b", "c"];
    let v: Vec<&str> = xs.iter().map(|x| *x).intersperse(", ").collect();
    let text: String = v.concat();
    assert_eq!(text, "a, , b, c".to_string());

    let ys = [0, 1, 2, 3];
    let mut it = ys[..0].iter().map(|x| *x).intersperse(1);
    assert!(it.next() == None);
}

#[test]
fn test_intersperse_size_hint() {
    let iter = std::iter::empty::<i32>().intersperse(0);
    assert_eq!(iter.size_hint(), (0, Some(0)));

    let xs = ["a", "", "b", "c"];
    let mut iter = xs.iter().map(|x| *x).intersperse(", ");
    assert_eq!(iter.size_hint(), (7, Some(7)));

    assert_eq!(iter.next(), Some("a"));
    assert_eq!(iter.size_hint(), (6, Some(6)));
    assert_eq!(iter.next(), Some(", "));
    assert_eq!(iter.size_hint(), (5, Some(5)));

    assert_eq!([].iter().intersperse(&()).size_hint(), (0, Some(0)));
}

#[test]
fn test_fold_specialization_intersperse() {
    let mut iter = (1..2).intersperse(0);
    iter.clone().for_each(|x| assert_eq!(Some(x), iter.next()));

    let mut iter = (1..3).intersperse(0);
    iter.clone().for_each(|x| assert_eq!(Some(x), iter.next()));

    let mut iter = (1..4).intersperse(0);
    iter.clone().for_each(|x| assert_eq!(Some(x), iter.next()));
}

#[test]
fn test_try_fold_specialization_intersperse_ok() {
    let mut iter = (1..2).intersperse(0);
    iter.clone().try_for_each(|x| {
        assert_eq!(Some(x), iter.next());
        Some(())
    });

    let mut iter = (1..3).intersperse(0);
    iter.clone().try_for_each(|x| {
        assert_eq!(Some(x), iter.next());
        Some(())
    });

    let mut iter = (1..4).intersperse(0);
    iter.clone().try_for_each(|x| {
        assert_eq!(Some(x), iter.next());
        Some(())
    });
}

#[test]
fn test_intersperse_with() {
    #[derive(PartialEq, Debug)]
    struct NotClone {
        u: u32,
    }
    let r = [NotClone { u: 0 }, NotClone { u: 1 }]
        .into_iter()
        .intersperse_with(|| NotClone { u: 2 })
        .collect::<Vec<_>>();
    assert_eq!(r, vec![NotClone { u: 0 }, NotClone { u: 2 }, NotClone { u: 1 }]);

    let mut ctr = 100;
    let separator = || {
        ctr *= 2;
        ctr
    };
    let r = (0..3).intersperse_with(separator).collect::<Vec<_>>();
    assert_eq!(r, vec![0, 200, 1, 400, 2]);
}

#[test]
fn test_intersperse_fold() {
    let v = (1..4).intersperse(9).fold(Vec::new(), |mut acc, x| {
        acc.push(x);
        acc
    });
    assert_eq!(v.as_slice(), [1, 9, 2, 9, 3]);

    let mut iter = (1..4).intersperse(9);
    assert_eq!(iter.next(), Some(1));
    let v = iter.fold(Vec::new(), |mut acc, x| {
        acc.push(x);
        acc
    });
    assert_eq!(v.as_slice(), [9, 2, 9, 3]);

    struct NoneAtStart(i32); // Produces: None, Some(2), Some(3), None, ...
    impl Iterator for NoneAtStart {
        type Item = i32;
        fn next(&mut self) -> Option<i32> {
            self.0 += 1;
            Some(self.0).filter(|i| i % 3 != 1)
        }
    }

    let v = NoneAtStart(0).intersperse(1000).fold(0, |a, b| a + b);
    assert_eq!(v, 0);
}

#[test]
fn test_intersperse_collect_string() {
    let contents = [1, 2, 3];

    let contents_string = contents
        .into_iter()
        .map(|id| id.to_string())
        .intersperse(", ".to_owned())
        .collect::<String>();
    assert_eq!(contents_string, "1, 2, 3");
}

#[test]
fn test_try_fold_specialization_intersperse_err() {
    let orig_iter = ["a", "b"].iter().copied().intersperse("-");

    // Abort after the first item.
    let mut iter = orig_iter.clone();
    iter.try_for_each(|_| None::<()>);
    assert_eq!(iter.next(), Some("-"));
    assert_eq!(iter.next(), Some("b"));
    assert_eq!(iter.next(), None);

    // Abort after the second item.
    let mut iter = orig_iter.clone();
    iter.try_for_each(|item| if item == "-" { None } else { Some(()) });
    assert_eq!(iter.next(), Some("b"));
    assert_eq!(iter.next(), None);

    // Abort after the third item.
    let mut iter = orig_iter.clone();
    iter.try_for_each(|item| if item == "b" { None } else { Some(()) });
    assert_eq!(iter.next(), None);
}
