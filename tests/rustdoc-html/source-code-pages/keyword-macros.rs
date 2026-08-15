// This test ensures that keywords which can be followed by values (and therefore `!`)
// are not considered as macros.
// This is a regression test for <https://github.com/rust-lang/rust/issues/148617>.

#![crate_name = "foo"]
#![feature(negative_impls)]

//@ has 'src/foo/keyword-macros.rs.html'

//@ has - '//*[@class="rust"]//*[@class="number"]' '2'
//@ has - '//*[@class="rust"]//*[@class="number"]' '0'
//@ has - '//*[@class="rust"]//*[@class="number"]' '1'
const ARR: [u8; 2] = [!0,! 1];

trait X {}

//@ has - '//*[@class="rust"]//*[@class="kw"]' 'impl'
impl !X for i32 {}

fn a() {
    //@ has - '//*[@class="rust"]//*[@class="kw"]' 'if'
    if! true{}
    //@ has - '//*[@class="rust"]//*[@class="kw"]' 'match'
    match !true { _ => {} }
    //@ has - '//*[@class="rust"]//*[@class="kw"]' 'while'
    let _ = while !true {
    //@ has - '//*[@class="rust"]//*[@class="kw"]' 'break'
        break !true;
    };
}
