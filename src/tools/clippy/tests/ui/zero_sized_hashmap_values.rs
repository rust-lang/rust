#![warn(clippy::zero_sized_map_values)]
use std::collections::HashMap;

const CONST_OK: Option<HashMap<String, usize>> = None;
const CONST_NOT_OK: Option<HashMap<String, ()>> = None;
//~^ zero_sized_map_values

static STATIC_OK: Option<HashMap<String, usize>> = None;
static STATIC_NOT_OK: Option<HashMap<String, ()>> = None;
//~^ zero_sized_map_values

type OkMap = HashMap<String, usize>;
type NotOkMap = HashMap<String, ()>;
//~^ zero_sized_map_values

enum TestEnum {
    Ok(HashMap<String, usize>),
    NotOk(HashMap<String, ()>),
    //~^ zero_sized_map_values
}

struct Test {
    ok: HashMap<String, usize>,
    not_ok: HashMap<String, ()>,
    //~^ zero_sized_map_values
    also_not_ok: Vec<HashMap<usize, ()>>,
    //~^ zero_sized_map_values
}

trait TestTrait {
    type Output;

    fn produce_output() -> Self::Output;

    fn weird_map(&self, map: HashMap<usize, ()>);
    //~^ zero_sized_map_values
}

impl Test {
    fn ok(&self) -> HashMap<String, usize> {
        todo!()
    }

    fn not_ok(&self) -> HashMap<String, ()> {
        //~^ zero_sized_map_values

        todo!()
    }
}

impl TestTrait for Test {
    type Output = HashMap<String, ()>;

    fn produce_output() -> Self::Output {
        todo!();
    }

    fn weird_map(&self, map: HashMap<usize, ()>) {
        todo!();
    }
}

fn test(map: HashMap<String, ()>, key: &str) -> HashMap<String, ()> {
    //~^ zero_sized_map_values
    //~| zero_sized_map_values

    todo!();
}

fn test2(map: HashMap<String, usize>, key: &str) -> HashMap<String, usize> {
    todo!();
}

fn main() {
    let _: HashMap<String, ()> = HashMap::new();
    //~^ zero_sized_map_values
    //~| zero_sized_map_values

    let _: HashMap<String, usize> = HashMap::new();

    let _: HashMap<_, _> = std::iter::empty::<(String, ())>().collect();
    //~^ zero_sized_map_values
}
