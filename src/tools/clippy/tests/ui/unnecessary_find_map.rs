#![allow(dead_code)]

fn main() {
    let _ = (0..4).find_map(|x| if x > 1 { Some(x) } else { None });
    //~^ unnecessary_find_map

    let _ = (0..4).find_map(|x| {
        //~^ unnecessary_find_map

        if x > 1 {
            return Some(x);
        };
        None
    });
    let _ = (0..4).find_map(|x| match x {
        //~^ unnecessary_find_map
        0 | 1 => None,
        _ => Some(x),
    });

    let _ = (0..4).find_map(|x| Some(x + 1));
    //~^ unnecessary_find_map

    let _ = (0..4).find_map(i32::checked_abs);
}

fn find_map_none_changes_item_type() -> Option<bool> {
    "".chars().find_map(|_| None)
}

fn issue11260() {
    let y = Some(1);
    let _x = std::iter::once(1).find_map(|n| (n > 1).then_some(n));
    //~^ unnecessary_find_map
    let _x = std::iter::once(1).find_map(|n| (n > 1).then_some(y)); // different option, so can't be just `.find()`
}
