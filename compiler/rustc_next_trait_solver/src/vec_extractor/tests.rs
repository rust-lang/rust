use crate::vec_extractor::Extractor;

#[test]
fn smoke() {
    let mut v = vec![0, 1, 2, 3];

    {
        let mut e = Extractor::new(&mut v);

        let entry = e.entry().unwrap();
        assert_eq!(entry.as_ref(), 0);
        entry.keep();

        let entry = e.entry().unwrap();
        assert_eq!(entry.as_ref(), 1);
        assert_eq!(entry.into_value(), 1);

        let entry = e.entry().unwrap();
        assert_eq!(entry.as_ref(), 2);
        let (two, hole) = entry.take();
        assert_eq!(two, 2);
        hole.fill(17);

        let mut entry = e.entry().unwrap();
        assert_eq!(entry.as_ref(), 3);
        *entry.as_mut() = 42;
        entry.keep();

        assert!(e.entry().is_none());
    }

    assert_eq!(v, [0, 17, 42]);
}

#[test]
fn all_keep() {
    let mut vec = (0..100).into_iter().collect::<Vec<_>>();

    {
        let mut e = Extractor::new(&mut vec);

        while let Some(entry) = e.entry() {
            drop(entry);
        }
    }

    assert_eq!(vec, []);
}

#[test]
fn all_keep2() {
    let mut vec = (0..100).into_iter().collect::<Vec<_>>();

    {
        let mut e = Extractor::new(&mut vec);

        while let Some(entry) = e.entry()
            && entry.as_ref() <= 25
        {
            drop(entry)
        }
    }

    assert_eq!(vec, (26..100).into_iter().collect::<Vec<_>>())
}

#[test]
fn all_take() {
    let mut vec = (0..100).into_iter().collect::<Vec<_>>();

    {
        let mut e = Extractor::new(&mut vec);

        while let Some(entry) = e.entry() {
            entry.take();
        }
    }

    assert_eq!(vec, []);
}

#[test]
fn take_n_drop_rest() {
    let mut vec = (0..100).into_iter().collect::<Vec<_>>();

    {
        let mut e = Extractor::new(&mut vec);

        while let Some(entry) = e.entry()
            && entry.as_ref() <= 25
        {
            entry.take();
        }

        e.drop_rest();
    }

    assert_eq!(vec, []);
}
