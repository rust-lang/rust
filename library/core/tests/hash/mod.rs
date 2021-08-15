mod sip;

use std::default::Default;
use std::hash::{BuildHasher, Hash, Hasher};
use std::rc::Rc;

struct MyHasher {
    hash: u64,
}

impl Default for MyHasher {
    fn default() -> MyHasher {
        MyHasher { hash: 0 }
    }
}

impl Hasher for MyHasher {
    fn write(&mut self, buf: &[u8]) {
        for byte in buf {
            self.hash += *byte as u64;
        }
    }
    fn finish(&self) -> u64 {
        self.hash
    }
}

#[test]
fn test_writer_hasher() {
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = MyHasher { hash: 0 };
        t.hash(&mut s);
        s.finish()
    }

    assert_eq!(hash(&()), 0);

    assert_eq!(hash(&5_u8), 5);
    assert_eq!(hash(&5_u16), 5);
    assert_eq!(hash(&5_u32), 5);
    assert_eq!(hash(&5_u64), 5);
    assert_eq!(hash(&5_usize), 5);

    assert_eq!(hash(&5_i8), 5);
    assert_eq!(hash(&5_i16), 5);
    assert_eq!(hash(&5_i32), 5);
    assert_eq!(hash(&5_i64), 5);
    assert_eq!(hash(&5_isize), 5);

    assert_eq!(hash(&false), 0);
    assert_eq!(hash(&true), 1);

    assert_eq!(hash(&'a'), 97);

    let s: &str = "a";
    assert_eq!(hash(&s), 97 + 0xFF);
    let s: Box<str> = String::from("a").into_boxed_str();
    assert_eq!(hash(&s), 97 + 0xFF);
    let s: Rc<&str> = Rc::new("a");
    assert_eq!(hash(&s), 97 + 0xFF);
    let cs: &[u8] = &[1, 2, 3];
    assert_eq!(hash(&cs), 9);
    let cs: Box<[u8]> = Box::new([1, 2, 3]);
    assert_eq!(hash(&cs), 9);
    let cs: Rc<[u8]> = Rc::new([1, 2, 3]);
    assert_eq!(hash(&cs), 9);

    let ptr = 5_usize as *const i32;
    assert_eq!(hash(&ptr), 5);

    let ptr = 5_usize as *mut i32;
    assert_eq!(hash(&ptr), 5);

    if cfg!(miri) {
        // Miri cannot hash pointers
        return;
    }

    let cs: &mut [u8] = &mut [1, 2, 3];
    let ptr = cs.as_ptr();
    let slice_ptr = cs as *const [u8];
    assert_eq!(hash(&slice_ptr), hash(&ptr) + cs.len() as u64);

    let slice_ptr = cs as *mut [u8];
    assert_eq!(hash(&slice_ptr), hash(&ptr) + cs.len() as u64);
}

struct Custom {
    hash: u64,
}
struct CustomHasher {
    output: u64,
}

impl Hasher for CustomHasher {
    fn finish(&self) -> u64 {
        self.output
    }
    fn write(&mut self, _: &[u8]) {
        panic!()
    }
    fn write_u64(&mut self, data: u64) {
        self.output = data;
    }
}

impl Default for CustomHasher {
    fn default() -> CustomHasher {
        CustomHasher { output: 0 }
    }
}

impl Hash for Custom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[test]
fn test_custom_state() {
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut c = CustomHasher { output: 0 };
        t.hash(&mut c);
        c.finish()
    }

    assert_eq!(hash(&Custom { hash: 5 }), 5);
}

// FIXME: Instantiated functions with i128 in the signature is not supported in Emscripten.
// See https://github.com/kripken/emscripten-fastcomp/issues/169
#[cfg(not(target_os = "emscripten"))]
#[test]
fn test_indirect_hasher() {
    let mut hasher = MyHasher { hash: 0 };
    {
        let mut indirect_hasher: &mut dyn Hasher = &mut hasher;
        5u32.hash(&mut indirect_hasher);
    }
    assert_eq!(hasher.hash, 5);
}

#[test]
fn test_build_hasher_object_safe() {
    use std::collections::hash_map::{DefaultHasher, RandomState};

    let _: &dyn BuildHasher<Hasher = DefaultHasher> = &RandomState::new();
}
