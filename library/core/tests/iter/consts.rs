#[test]
fn const_manual_iter() {
    struct S(bool);

    impl const Iterator for S {
        type Item = ();

        fn next(&mut self) -> Option<Self::Item> {
            if self.0 == false {
                self.0 = true;
                Some(())
            } else {
                None
            }
        }
    }
    const {
        let mut val = S(false);
        assert!(val.next().is_some());
        assert!(val.next().is_none());
        assert!(val.next().is_none());
    }
}

#[test]
fn const_range() {
    const {
        let mut arr = [0; 3];
        for i in 0..arr.len() {
            arr[i] = i;
        }
        assert!(arr[0] == 0);
        assert!(arr[1] == 1);
        assert!(arr[2] == 2);
    }
}
