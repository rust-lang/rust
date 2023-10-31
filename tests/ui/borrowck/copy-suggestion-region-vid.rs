//@run-rustfix
pub struct DataStruct();

pub struct HelperStruct<'n> {
    pub helpers: [Vec<&'n i64>; 2],
    pub is_empty: bool,
}

impl DataStruct {
    pub fn f(&self) -> HelperStruct {
        let helpers = [vec![], vec![]];

        HelperStruct { helpers, is_empty: helpers[0].is_empty() }
        //~^ ERROR borrow of moved value
    }
}

fn main() {}
