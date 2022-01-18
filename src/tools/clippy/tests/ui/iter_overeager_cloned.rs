// run-rustfix
#![warn(clippy::iter_overeager_cloned, clippy::redundant_clone, clippy::filter_next)]

fn main() {
    let vec = vec!["1".to_string(), "2".to_string(), "3".to_string()];

    let _: Option<String> = vec.iter().cloned().last();

    let _: Option<String> = vec.iter().chain(vec.iter()).cloned().next();

    let _: usize = vec.iter().filter(|x| x == &"2").cloned().count();

    let _: Vec<_> = vec.iter().cloned().take(2).collect();

    let _: Vec<_> = vec.iter().cloned().skip(2).collect();

    let _ = vec.iter().filter(|x| x == &"2").cloned().nth(2);

    let _ = [Some(Some("str".to_string())), Some(Some("str".to_string()))]
        .iter()
        .cloned()
        .flatten();

    // Not implemented yet
    let _ = vec.iter().cloned().filter(|x| x.starts_with('2'));

    // Not implemented yet
    let _ = vec.iter().cloned().map(|x| x.len());

    // This would fail if changed.
    let _ = vec.iter().cloned().map(|x| x + "2");

    // Not implemented yet
    let _ = vec.iter().cloned().find(|x| x == "2");

    // Not implemented yet
    let _ = vec.iter().cloned().for_each(|x| assert!(!x.is_empty()));

    // Not implemented yet
    let _ = vec.iter().cloned().all(|x| x.len() == 1);

    // Not implemented yet
    let _ = vec.iter().cloned().any(|x| x.len() == 1);

    // Should probably stay as it is.
    let _ = [0, 1, 2, 3, 4].iter().cloned().take(10);
}
