use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Mutex};

fn main() {
    my_very_long_function_name_with_lots_of_args(); //~ ERROR E0061
}

fn my_very_long_function_name_with_lots_of_args(
    _first_long_param: bool,
    _second_long_param: HashMap<Arc<String>, Vec<BTreeSet<usize>>>,
    _third_long_param: usize,
    _fourth_long_param: String,
    _fifth_long_param: Mutex<Option<VecDeque<Arc<String>>>>,
) {
}
