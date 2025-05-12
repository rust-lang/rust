//@needs-asm-support

#![allow(unused, clippy::needless_pass_by_value, clippy::collapsible_if)]
#![warn(clippy::map_entry)]

use std::arch::asm;
use std::collections::HashMap;
use std::hash::Hash;

macro_rules! m {
    ($e:expr) => {{ $e }};
}

macro_rules! insert {
    ($map:expr, $key:expr, $val:expr) => {
        $map.insert($key, $val)
    };
}

fn foo() {}

fn hash_map<K: Eq + Hash + Copy, V: Copy>(m: &mut HashMap<K, V>, m2: &mut HashMap<K, V>, k: K, k2: K, v: V, v2: V) {
    // or_insert(v)
    if !m.contains_key(&k) {
        //~^ map_entry
        m.insert(k, v);
    }

    // semicolon on insert, use or_insert_with(..)
    if !m.contains_key(&k) {
        //~^ map_entry
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
        }
    }

    // semicolon on if, use or_insert_with(..)
    if !m.contains_key(&k) {
        //~^ map_entry
        if true {
            m.insert(k, v)
        } else {
            m.insert(k, v2)
        };
    }

    // early return, use if let
    if !m.contains_key(&k) {
        //~^ map_entry
        if true {
            m.insert(k, v);
        } else {
            m.insert(k, v2);
            return;
        }
    }

    // use or_insert_with(..)
    if !m.contains_key(&k) {
        //~^ map_entry
        foo();
        m.insert(k, v);
    }

    // semicolon on insert and match, use or_insert_with(..)
    if !m.contains_key(&k) {
        //~^ map_entry
        match 0 {
            1 if true => {
                m.insert(k, v);
            },
            _ => {
                m.insert(k, v2);
            },
        };
    }

    // one branch doesn't insert, use if let
    if !m.contains_key(&k) {
        //~^ map_entry
        match 0 {
            0 => foo(),
            _ => {
                m.insert(k, v2);
            },
        };
    }

    // use or_insert_with
    if !m.contains_key(&k) {
        //~^ map_entry
        foo();
        match 0 {
            0 if false => {
                m.insert(k, v);
            },
            1 => {
                foo();
                m.insert(k, v);
            },
            2 | 3 => {
                for _ in 0..2 {
                    foo();
                }
                if true {
                    m.insert(k, v);
                } else {
                    m.insert(k, v2);
                };
            },
            _ => {
                m.insert(k, v2);
            },
        }
    }

    // ok, insert in loop
    if !m.contains_key(&k) {
        for _ in 0..2 {
            m.insert(k, v);
        }
    }

    // macro_expansion test, use or_insert(..)
    if !m.contains_key(&m!(k)) {
        //~^ map_entry
        m.insert(m!(k), m!(v));
    }

    // ok, map used before insertion
    if !m.contains_key(&k) {
        let _ = m.len();
        m.insert(k, v);
    }

    // ok, inline asm
    if !m.contains_key(&k) {
        unsafe { asm!("nop") }
        m.insert(k, v);
    }

    // ok, different keys.
    if !m.contains_key(&k) {
        m.insert(k2, v);
    }

    // ok, different maps
    if !m.contains_key(&k) {
        m2.insert(k, v);
    }

    // ok, insert in macro
    if !m.contains_key(&k) {
        insert!(m, k, v);
    }

    // or_insert_with. Partial move of a local declared in the closure is ok.
    if !m.contains_key(&k) {
        //~^ map_entry
        let x = (String::new(), String::new());
        let _ = x.0;
        m.insert(k, v);
    }
}

// Issue 10331
// do not suggest a bad expansion because the compiler unrolls the first
// occurrence of the loop
pub fn issue_10331() {
    let mut m = HashMap::new();
    let mut i = 0;
    let mut x = 0;
    while !m.contains_key(&x) {
        m.insert(x, i);
        i += 1;
        x += 1;
    }
}

/// Issue 11935
/// Do not suggest using entries if the map is used inside the `insert` expression.
pub fn issue_11935() {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    if !counts.contains_key(&1) {
        counts.insert(1, 1);
    } else {
        counts.insert(1, counts.get(&1).unwrap() + 1);
    }
}

fn issue12489(map: &mut HashMap<u64, u64>) -> Option<()> {
    if !map.contains_key(&1) {
        //~^ map_entry
        let Some(1) = Some(2) else {
            return None;
        };
        map.insert(1, 42);
    }
    Some(())
}

mod issue13934 {
    use std::collections::HashMap;

    struct Member {}

    pub struct Foo {
        members: HashMap<u8, Member>,
    }

    impl Foo {
        pub fn should_also_not_cause_lint(&mut self, input: u8) {
            if self.members.contains_key(&input) {
                todo!();
            } else {
                self.other();
                self.members.insert(input, Member {});
            }
        }

        fn other(&self) {}
    }
}

fn issue11976() {
    let mut hashmap = std::collections::HashMap::new();
    if !hashmap.contains_key(&0) {
        let _ = || hashmap.get(&0);
        hashmap.insert(0, 0);
    }
}

mod issue14449 {
    use std::collections::BTreeMap;

    pub struct Meow {
        map: BTreeMap<String, String>,
    }

    impl Meow {
        fn pet(&self, _key: &str, _v: u32) -> u32 {
            42
        }
    }

    pub fn f(meow: &Meow, x: String) {
        if meow.map.contains_key(&x) {
            let _ = meow.pet(&x, 1);
        } else {
            let _ = meow.pet(&x, 0);
        }
    }
}

fn main() {}
