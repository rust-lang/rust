//@edition: 2021
#![warn(clippy::search_is_some)]

fn main() {
    fn ref_bindings() {
        let _ = [&(&1, 2), &(&3, 4), &(&5, 4)].iter().find(|(&x, y)| x == *y).is_none();
        //~^ search_is_some
        let _ = [&(&1, 2), &(&3, 4), &(&5, 4)].iter().find(|&(&x, y)| x == *y).is_none();
        //~^ search_is_some
        let _ = [&(&1, 2), &(&3, 4), &(&5, 4)]
            //~^ search_is_some
            .iter()
            .find(|&&&(&x, ref y)| x == *y)
            .is_none();
    }
}
