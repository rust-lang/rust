#![warn(clippy::zero_sized_map_values)]
use std::collections::BTreeMap;

const CONST_OK: Option<BTreeMap<String, usize>> = None;
const CONST_NOT_OK: Option<BTreeMap<String, ()>> = None;
//~^ zero_sized_map_values

static STATIC_OK: Option<BTreeMap<String, usize>> = None;
static STATIC_NOT_OK: Option<BTreeMap<String, ()>> = None;
//~^ zero_sized_map_values

type OkMap = BTreeMap<String, usize>;
type NotOkMap = BTreeMap<String, ()>;
//~^ zero_sized_map_values

enum TestEnum {
    Ok(BTreeMap<String, usize>),
    NotOk(BTreeMap<String, ()>),
    //~^ zero_sized_map_values
}

struct Test {
    ok: BTreeMap<String, usize>,
    not_ok: BTreeMap<String, ()>,
    //~^ zero_sized_map_values
    also_not_ok: Vec<BTreeMap<usize, ()>>,
    //~^ zero_sized_map_values
}

trait TestTrait {
    type Output;

    fn produce_output() -> Self::Output;

    fn weird_map(&self, map: BTreeMap<usize, ()>);
    //~^ zero_sized_map_values
}

impl Test {
    fn ok(&self) -> BTreeMap<String, usize> {
        todo!()
    }

    fn not_ok(&self) -> BTreeMap<String, ()> {
        //~^ zero_sized_map_values

        todo!()
    }
}

impl TestTrait for Test {
    type Output = BTreeMap<String, ()>;

    fn produce_output() -> Self::Output {
        todo!();
    }

    fn weird_map(&self, map: BTreeMap<usize, ()>) {
        todo!();
    }
}

fn test(map: BTreeMap<String, ()>, key: &str) -> BTreeMap<String, ()> {
    //~^ zero_sized_map_values
    //~| zero_sized_map_values

    todo!();
}

fn test2(map: BTreeMap<String, usize>, key: &str) -> BTreeMap<String, usize> {
    todo!();
}

fn main() {
    let _: BTreeMap<String, ()> = BTreeMap::new();
    //~^ zero_sized_map_values
    //~| zero_sized_map_values

    let _: BTreeMap<String, usize> = BTreeMap::new();

    let _: BTreeMap<_, _> = std::iter::empty::<(String, ())>().collect();
    //~^ zero_sized_map_values
}
