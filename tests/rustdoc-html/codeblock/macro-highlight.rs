/// ```
/// foo!();
/// bar![];
/// baz!{}
/// whitespace  !  (  )  ;
/// std::println!("scientists reveal the hi");
/// core::panic!();
/// outer!(inner!());
/// macro_rules! abc {}
/// union!();
/// raw!();
/// auto!();
/// r#async!();
/// ```
pub fn passing() {}

//@ has macro_highlight/fn.passing.html
//@ has - '//*[@class="macro"]' 'foo!'
//@ has - '//*[@class="macro"]' 'bar!'
//@ has - '//*[@class="macro"]' 'baz!'
//@ has - '//*[@class="macro"]' 'whitespace'
//@ has - '//*[@class="macro"]' 'println!'
//@ has - '//*[@class="macro"]' 'panic!'
//@ has - '//*[@class="macro"]' 'outer!'
//@ has - '//*[@class="macro"]' 'inner!'
//@ has - '//*[@class="macro"]' 'union!'
//@ has - '//*[@class="macro"]' 'raw!'
//@ has - '//*[@class="macro"]' 'auto!'
//@ has - '//*[@class="macro"]' 'r#async!'

/// ```
/// x!= y;
/// impl SomeTrait for! {}
/// impl! {}
/// if! true {}
/// while! false {}
/// impl! Send for u32 {}
/// ```
pub fn failing() {}

//@ has macro_highlight/fn.failing.html
//@ count - '//*[@class="macro"]' 0
//@ !has - '//*[@class="macro"]' 'x'
//@ !has - '//*[@class="macro"]' 'for'
//@ !has - '//*[@class="macro"]' 'impl'
//@ !has - '//*[@class="macro"]' 'if'
//@ !has - '//*[@class="macro"]' 'while'
