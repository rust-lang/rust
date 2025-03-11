#![allow(clippy::question_mark, unused)]
#![warn(clippy::filter_map_bool_then)]
//@no-rustfix

fn issue11617() {
    let mut x: Vec<usize> = vec![0; 10];
    let _ = (0..x.len()).zip(x.clone().iter()).filter_map(|(i, v)| {
        //~^ filter_map_bool_then
        (x[i] != *v).then(|| {
            x[i] = i;
            i
        })
    });
}

mod issue14368 {

    fn do_something(_: ()) -> bool {
        true
    }

    fn option_with_early_return(x: &[Option<bool>]) {
        let _ = x.iter().filter_map(|&x| x?.then(|| do_something(())));
        //~^ filter_map_bool_then
        let _ = x
            .iter()
            .filter_map(|&x| if let Some(x) = x { x } else { return None }.then(|| do_something(())));
        //~^ filter_map_bool_then
        let _ = x.iter().filter_map(|&x| {
            //~^ filter_map_bool_then
            match x {
                Some(x) => x,
                None => return None,
            }
            .then(|| do_something(()))
        });
    }

    #[derive(Copy, Clone)]
    enum Foo {
        One(bool),
        Two,
        Three(Option<i32>),
    }

    fn nested_type_with_early_return(x: &[Foo]) {
        let _ = x.iter().filter_map(|&x| {
            //~^ filter_map_bool_then
            match x {
                Foo::One(x) => x,
                Foo::Two => return None,
                Foo::Three(inner) => {
                    if inner? == 0 {
                        return Some(false);
                    } else {
                        true
                    }
                },
            }
            .then(|| do_something(()))
        });
    }
}
