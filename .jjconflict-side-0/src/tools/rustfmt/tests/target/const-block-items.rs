#![feature(const_block_items)]

const { assert!(true) }

#[cfg(false)]
const { assert!(false) }

#[cfg(false)]
// foo
const {
    // bar
    assert!(false)
    // baz
} // 123

#[expect(unused)]
const {
    let a = 1;
    assert!(true);
}
