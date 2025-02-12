#![warn(clippy::iter_filter_is_some)]
#![allow(
    clippy::map_identity,
    clippy::result_filter_map,
    clippy::needless_borrow,
    clippy::option_filter_map,
    clippy::redundant_closure,
    clippy::unnecessary_get_then_check
)]

use std::collections::HashMap;

fn main() {
    {
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(Option::is_some);
        //~^ iter_filter_is_some

        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|a| a.is_some());
        //~^ iter_filter_is_some

        #[rustfmt::skip]
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|o| { o.is_some() });
        //~^ iter_filter_is_some
    }

    {
        let _ = vec![Some(1), None, Some(3)]
            .into_iter()
            .filter(std::option::Option::is_some);
        //~^ iter_filter_is_some

        let _ = vec![Some(1), None, Some(3)]
            .into_iter()
            .filter(|a| std::option::Option::is_some(a));
        //~^ iter_filter_is_some

        #[rustfmt::skip]
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|a| { std::option::Option::is_some(a) });
        //~^ iter_filter_is_some
    }

    {
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|&a| a.is_some());
        //~^ iter_filter_is_some

        #[rustfmt::skip]
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|&o| { o.is_some() });
        //~^ iter_filter_is_some
    }

    {
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|ref a| a.is_some());
        //~^ iter_filter_is_some

        #[rustfmt::skip]
        let _ = vec![Some(1), None, Some(3)].into_iter().filter(|ref o| { o.is_some() });
        //~^ iter_filter_is_some
    }
}

fn avoid_linting_when_filter_has_side_effects() {
    // Don't lint below
    let mut counter = 0;
    let _ = vec![Some(1), None, Some(3)].into_iter().filter(|o| {
        counter += 1;
        o.is_some()
    });
}

fn avoid_linting_when_commented() {
    let _ = vec![Some(1), None, Some(3)].into_iter().filter(|o| {
        // Roses are red,
        // Violets are blue,
        // `Err` is not an `Option`,
        // and this doesn't ryme
        o.is_some()
    });
}

fn ice_12058() {
    // check that checking the parent node doesn't cause an ICE
    // by indexing the parameters of a closure without parameters
    Some(1).or_else(|| {
        vec![Some(1), None, Some(3)].into_iter().filter(|z| *z != Some(2));
        None
    });
}

fn avoid_linting_map() {
    // should not lint
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        .filter(|o| o.is_some())
        .map(|o| o.unwrap());

    // should not lint
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        .filter(|o| o.is_some())
        .map(|o| o);
}

fn avoid_false_positive_due_to_is_some_and_iterator_impl() {
    #[derive(Default, Clone)]
    struct Foo {}

    impl Foo {
        fn is_some(&self) -> bool {
            true
        }
    }

    impl Iterator for Foo {
        type Item = Foo;
        fn next(&mut self) -> Option<Self::Item> {
            Some(Foo::default())
        }
    }

    let data = vec![Foo::default()];
    // should not lint
    let _ = data.clone().into_iter().filter(Foo::is_some);
    // should not lint
    let _ = data.clone().into_iter().filter(|f| f.is_some());
}

fn avoid_false_positive_due_to_is_some_and_into_iterator_impl() {
    #[derive(Default, Clone)]
    struct Foo {}

    impl Foo {
        fn is_some(&self) -> bool {
            true
        }
    }

    let data = vec![Foo::default()];
    // should not lint
    let _ = data.clone().into_iter().filter(Foo::is_some);
    // should not lint
    let _ = data.clone().into_iter().filter(|f| f.is_some());
}

fn avoid_unpack_fp() {
    let _ = vec![(Some(1), None), (None, Some(3))]
        .into_iter()
        // should not lint
        .filter(|(a, _)| a.is_some());
    let _ = vec![(Some(1), None), (None, Some(3))]
        .into_iter()
        // should not lint
        .filter(|(a, _)| a.is_some())
        .collect::<Vec<_>>();

    let m = HashMap::from([(1, 1)]);
    let _ = vec![1, 2, 4].into_iter().filter(|a| m.get(a).is_some());
    // should not lint
}

fn avoid_fp_for_external() {
    let value = HashMap::from([(1, 1)]);
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| value.get(&1).is_some());

    let value = Option::Some(1);
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| value.is_some());
}

fn avoid_fp_for_trivial() {
    let value = HashMap::from([(1, 1)]);
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| Some(1).is_some());
    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| None::<i32>.is_some());
}

fn avoid_false_positive_due_to_method_name() {
    fn is_some(x: &Option<i32>) -> bool {
        x.is_some()
    }

    vec![Some(1), None, Some(3)].into_iter().filter(is_some);
    // should not lint
}

fn avoid_fp_due_to_trait_type() {
    struct Foo {
        bar: i32,
    }
    impl Foo {
        fn is_some(obj: &Option<i32>) -> bool {
            obj.is_some()
        }
    }
    vec![Some(1), None, Some(3)].into_iter().filter(Foo::is_some);
    // should not lint
}

fn avoid_fp_with_call_to_outside_var() {
    let outside = Some(1);

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| outside.is_some());

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| Option::is_some(&outside));

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| std::option::Option::is_some(&outside));
}

fn avoid_fp_with_call_to_outside_var_mix_match_types() {
    let outside: Result<i32, ()> = Ok(1);

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| outside.is_ok());

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| Result::is_ok(&outside));

    let _ = vec![Some(1), None, Some(3)]
        .into_iter()
        // should not lint
        .filter(|o| std::result::Result::is_ok(&outside));
}
