// run-pass
#![allow(dead_code)]
#![feature(box_syntax)]

#[derive(Copy, Clone)]
struct LM { resize_at: usize, size: usize }

enum HashMap<K,V> {
    HashMap_(LM, Vec<(K,V)>)
}

fn linear_map<K,V>() -> HashMap<K,V> {
    HashMap::HashMap_(LM{
        resize_at: 32,
        size: 0}, Vec::new())
}

impl<K,V> HashMap<K,V> {
    pub fn len(&mut self) -> usize {
        match *self {
            HashMap::HashMap_(ref l, _) => l.size
        }
    }
}

pub fn main() {
    let mut m: Box<_> = box linear_map::<(),()>();
    assert_eq!(m.len(), 0);
}
