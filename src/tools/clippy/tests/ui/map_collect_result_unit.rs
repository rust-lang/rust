#![warn(clippy::map_collect_result_unit)]

fn main() {
    {
        let _ = (0..3).map(|t| Err(t + 1)).collect::<Result<(), _>>();
        //~^ map_collect_result_unit
        let _: Result<(), _> = (0..3).map(|t| Err(t + 1)).collect();
        //~^ map_collect_result_unit

        let _ = (0..3).try_for_each(|t| Err(t + 1));
    }
}

fn _ignore() {
    let _ = (0..3).map(|t| Err(t + 1)).collect::<Result<Vec<i32>, _>>();
    let _ = (0..3).map(|t| Err(t + 1)).collect::<Vec<Result<(), _>>>();
}
