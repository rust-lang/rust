#![warn(clippy::iter_overeager_cloned, clippy::redundant_clone, clippy::filter_next)]
#![allow(
    dead_code,
    clippy::let_unit_value,
    clippy::useless_vec,
    clippy::double_ended_iterator_last
)]

fn main() {
    let vec = vec!["1".to_string(), "2".to_string(), "3".to_string()];

    let _: Option<String> = vec.iter().cloned().last();
    //~^ iter_overeager_cloned

    let _: Option<String> = vec.iter().chain(vec.iter()).cloned().next();
    //~^ iter_overeager_cloned

    let _: usize = vec.iter().filter(|x| x == &"2").cloned().count();
    //~^ redundant_clone

    let _: Vec<_> = vec.iter().cloned().take(2).collect();
    //~^ iter_overeager_cloned

    let _: Vec<_> = vec.iter().cloned().skip(2).collect();
    //~^ iter_overeager_cloned

    let _ = vec.iter().filter(|x| x == &"2").cloned().nth(2);
    //~^ iter_overeager_cloned

    let _ = [Some(Some("str".to_string())), Some(Some("str".to_string()))]
        //~^ iter_overeager_cloned
        .iter()
        .cloned()
        .flatten();

    let _ = vec.iter().cloned().filter(|x| x.starts_with('2'));
    //~^ iter_overeager_cloned

    let _ = vec.iter().cloned().find(|x| x == "2");
    //~^ iter_overeager_cloned

    {
        let f = |x: &String| x.starts_with('2');
        let _ = vec.iter().cloned().filter(f);
        //~^ iter_overeager_cloned
        let _ = vec.iter().cloned().find(f);
        //~^ iter_overeager_cloned
    }

    {
        let vec: Vec<(String, String)> = vec![];
        let f = move |x: &(String, String)| x.0.starts_with('2');
        let _ = vec.iter().cloned().filter(f);
        //~^ iter_overeager_cloned
        let _ = vec.iter().cloned().find(f);
        //~^ iter_overeager_cloned
    }

    fn test_move<'a>(
        iter: impl Iterator<Item = &'a (&'a u32, String)> + 'a,
        target: String,
    ) -> impl Iterator<Item = (&'a u32, String)> + 'a {
        iter.cloned().filter(move |&(&a, ref b)| a == 1 && b == &target)
        //~^ iter_overeager_cloned
    }

    {
        #[derive(Clone)]
        struct S<'a> {
            a: &'a u32,
            b: String,
        }

        fn bar<'a>(iter: impl Iterator<Item = &'a S<'a>> + 'a, target: String) -> impl Iterator<Item = S<'a>> + 'a {
            iter.cloned().filter(move |S { a, b }| **a == 1 && b == &target)
            //~^ iter_overeager_cloned
        }
    }

    let _ = vec.iter().cloned().map(|x| x.len());
    //~^ redundant_clone

    // This would fail if changed.
    let _ = vec.iter().cloned().map(|x| x + "2");

    let _ = vec.iter().cloned().for_each(|x| assert!(!x.is_empty()));
    //~^ redundant_clone

    let _ = vec.iter().cloned().all(|x| x.len() == 1);
    //~^ redundant_clone

    let _ = vec.iter().cloned().any(|x| x.len() == 1);
    //~^ redundant_clone

    // Should probably stay as it is.
    let _ = [0, 1, 2, 3, 4].iter().cloned().take(10);

    // `&Range<_>` doesn't implement `IntoIterator`
    let _ = [0..1, 2..5].iter().cloned().flatten();
}

// #8527
fn cloned_flatten(x: Option<&Option<String>>) -> Option<String> {
    x.cloned().flatten()
}
