pub trait MyTrait {
    fn method_on_mytrait() {}
}

pub struct MyStruct;

impl MyStruct {
    pub fn method_on_mystruct() {}
}

// @has typedef/type.MyAlias.html
// @has - '//*[@class="impl"]//code' 'impl MyAlias'
// @has - '//*[@class="impl"]//code' 'impl MyTrait for MyAlias'
// @has - 'Alias docstring'
// @has - '//*[@class="sidebar"]//p[@class="location"]' 'Type Definition MyAlias'
// @has - '//*[@class="sidebar"]//a[@href="#methods"]' 'Methods'
// @has - '//*[@class="sidebar"]//a[@href="#implementations"]' 'Trait Implementations'
/// Alias docstring
pub type MyAlias = MyStruct;

impl MyAlias {
    pub fn method_on_myalias() {}
}

impl MyTrait for MyAlias {}
