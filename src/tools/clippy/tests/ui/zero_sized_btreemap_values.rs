#![warn(clippy::zero_sized_map_values)]
use std::collections::BTreeMap;

const CONST_OK: Option<BTreeMap<String, usize>> = None;
const CONST_NOT_OK: Option<BTreeMap<String, ()>> = None;

static STATIC_OK: Option<BTreeMap<String, usize>> = None;
static STATIC_NOT_OK: Option<BTreeMap<String, ()>> = None;

type OkMap = BTreeMap<String, usize>;
type NotOkMap = BTreeMap<String, ()>;

enum TestEnum {
    Ok(BTreeMap<String, usize>),
    NotOk(BTreeMap<String, ()>),
}

struct Test {
    ok: BTreeMap<String, usize>,
    not_ok: BTreeMap<String, ()>,

    also_not_ok: Vec<BTreeMap<usize, ()>>,
}

trait TestTrait {
    type Output;

    fn produce_output() -> Self::Output;

    fn weird_map(&self, map: BTreeMap<usize, ()>);
}

impl Test {
    fn ok(&self) -> BTreeMap<String, usize> {
        todo!()
    }

    fn not_ok(&self) -> BTreeMap<String, ()> {
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
    todo!();
}

fn test2(map: BTreeMap<String, usize>, key: &str) -> BTreeMap<String, usize> {
    todo!();
}

fn main() {
    let _: BTreeMap<String, ()> = BTreeMap::new();
    let _: BTreeMap<String, usize> = BTreeMap::new();

    let _: BTreeMap<_, _> = std::iter::empty::<(String, ())>().collect();
}
