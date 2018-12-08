#![cfg(feature = "serde")]

extern crate hashbrown;
extern crate serde_test;

use hashbrown::{HashMap, HashSet};
use serde_test::{assert_tokens, Token};

#[test]
fn map_serde_tokens_empty() {
    let map = HashMap::<char, u32>::new();

    assert_tokens(&map, &[Token::Map { len: Some(0) }, Token::MapEnd]);
}

#[test]
fn map_serde_tokens() {
    let mut map = HashMap::new();
    map.insert('b', 20);
    map.insert('a', 10);
    map.insert('c', 30);

    assert_tokens(
        &map,
        &[
            Token::Map { len: Some(3) },
            Token::Char('a'),
            Token::I32(10),
            Token::Char('b'),
            Token::I32(20),
            Token::Char('c'),
            Token::I32(30),
            Token::MapEnd,
        ],
    );
}

#[test]
fn set_serde_tokens_empty() {
    let set = HashSet::<u32>::new();

    assert_tokens(&set, &[Token::Seq { len: Some(0) }, Token::SeqEnd]);
}

#[test]
fn set_serde_tokens() {
    let mut set = HashSet::new();
    set.insert(20);
    set.insert(10);
    set.insert(30);

    assert_tokens(
        &set,
        &[
            Token::Seq { len: Some(3) },
            Token::I32(20),
            Token::I32(10),
            Token::I32(30),
            Token::SeqEnd,
        ],
    );
}
