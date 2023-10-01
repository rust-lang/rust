#![warn(clippy::zero_sized_map_values)]
use std::collections::HashMap;

const CONST_OK: Option<HashMap<String, usize>> = None;
const CONST_NOT_OK: Option<HashMap<String, ()>> = None;
//~^ ERROR: map with zero-sized value type

static STATIC_OK: Option<HashMap<String, usize>> = None;
static STATIC_NOT_OK: Option<HashMap<String, ()>> = None;
//~^ ERROR: map with zero-sized value type

type OkMap = HashMap<String, usize>;
type NotOkMap = HashMap<String, ()>;
//~^ ERROR: map with zero-sized value type

enum TestEnum {
    Ok(HashMap<String, usize>),
    NotOk(HashMap<String, ()>),
    //~^ ERROR: map with zero-sized value type
}

struct Test {
    ok: HashMap<String, usize>,
    not_ok: HashMap<String, ()>,
    //~^ ERROR: map with zero-sized value type
    also_not_ok: Vec<HashMap<usize, ()>>,
    //~^ ERROR: map with zero-sized value type
}

trait TestTrait {
    type Output;

    fn produce_output() -> Self::Output;

    fn weird_map(&self, map: HashMap<usize, ()>);
    //~^ ERROR: map with zero-sized value type
}

impl Test {
    fn ok(&self) -> HashMap<String, usize> {
        todo!()
    }

    fn not_ok(&self) -> HashMap<String, ()> {
        //~^ ERROR: map with zero-sized value type
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
    //~^ ERROR: map with zero-sized value type
    //~| ERROR: map with zero-sized value type
    todo!();
}

fn test2(map: HashMap<String, usize>, key: &str) -> HashMap<String, usize> {
    todo!();
}

fn main() {
    let _: HashMap<String, ()> = HashMap::new();
    //~^ ERROR: map with zero-sized value type
    //~| ERROR: map with zero-sized value type
    let _: HashMap<String, usize> = HashMap::new();

    let _: HashMap<_, _> = std::iter::empty::<(String, ())>().collect();
    //~^ ERROR: map with zero-sized value type
}
