fn main() {
    let _ = (0..4).filter_map(|x| if x > 1 { Some(x) } else { None });
    let _ = (0..4).filter_map(|x| {
        if x > 1 {
            return Some(x);
        };
        None
    });
    let _ = (0..4).filter_map(|x| match x {
        0 | 1 => None,
        _ => Some(x),
    });

    let _ = (0..4).filter_map(|x| Some(x + 1));

    let _ = (0..4).filter_map(i32::checked_abs);
}

fn filter_map_none_changes_item_type() -> impl Iterator<Item = bool> {
    "".chars().filter_map(|_| None)
}
