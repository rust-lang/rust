pub trait MyTrait {
    fn method_on_mytrait() {}
}

pub struct MyStruct;

impl MyStruct {
    pub fn method_on_mystruct() {}
}

//@ has typedef/type.MyAlias.html
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'impl MyAlias'
//@ has - '//*[@class="impl"]//h3[@class="code-header"]' 'impl MyTrait for MyAlias'
//@ hasraw - 'Alias docstring'
//@ has - '//*[@class="sidebar"]//*[@class="location"]' 'MyAlias'
//@ has - '//*[@class="sidebar"]//a[@href="#implementations"]' 'Methods'
//@ has - '//*[@class="sidebar"]//a[@href="#trait-implementations"]' 'Trait Implementations'
/// Alias docstring
pub type MyAlias = MyStruct;

impl MyAlias {
    pub fn method_on_myalias() {}
}

impl MyTrait for MyAlias {}
