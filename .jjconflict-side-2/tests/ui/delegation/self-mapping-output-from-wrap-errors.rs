#![feature(fn_delegation)]

mod pin_box_self {
    use std::pin::Pin;

    trait MyAdd {
        fn add(self, other: Self) -> Pin<Box<Self>>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Pin<Box<usize>> {
            Pin::new(Box::new(self + other))
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Pin<Box<usize>>);

    reuse impl MyAdd for W {
    //~^ ERROR: the trait bound `Pin<Box<pin_box_self::W>>: From<pin_box_self::W>` is not satisfied
        *self.0
    }
}

mod many_froms {
    use std::sync::Arc;
    use std::rc::Rc;

    trait MyAdd {
        fn add(self, other: Self) -> Box<Box<Box<Arc<Box<Rc<Self>>>>>>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Box<Box<Box<Arc<Box<Rc<usize>>>>>> {
            Box::new(Box::new(Box::new(Arc::new(Box::new(Rc::new(self + other))))))
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<Box<Box<Arc<Box<Rc<usize>>>>>>);

    reuse impl MyAdd for W {
    //~^ ERROR: the trait bound `Box<Box<Box<Arc<Box<Rc<many_froms::W>>>>>>: From<many_froms::W>` is not satisfied
        ******self.0
    }
}

mod many_froms_2 {
    use std::sync::Arc;
    use std::rc::Rc;

    trait MyAdd {
        fn add(self, other: Self) -> Box<Arc<Rc<Box<Rc<Self>>>>>;
    }

    impl MyAdd for usize {
        fn add(self, other: usize) -> Box<Arc<Rc<Box<Rc<usize>>>>> {
            Box::new(Arc::new(Rc::new(Box::new(Rc::new(self + other)))))
        }
    }

    #[derive(Eq, PartialEq, Debug)]
    struct W(Box<Arc<Rc<Box<Rc<usize>>>>>);

    reuse impl MyAdd for W {
    //~^ ERROR: the trait bound `Box<Arc<Rc<Box<Rc<many_froms_2::W>>>>>: From<many_froms_2::W>` is not satisfied
        *****self.0
    }
}

fn main() {
}
