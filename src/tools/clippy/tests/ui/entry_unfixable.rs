#![allow(clippy::needless_pass_by_value, clippy::collapsible_if)]
#![warn(clippy::map_entry)]

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

mod issue13306 {
    use std::collections::HashMap;

    struct Env {
        enclosing: Option<Box<Env>>,
        values: HashMap<String, usize>,
    }

    impl Env {
        fn assign(&mut self, name: String, value: usize) -> bool {
            if !self.values.contains_key(&name) {
                //~^ map_entry
                self.values.insert(name, value);
                true
            } else if let Some(enclosing) = &mut self.enclosing {
                enclosing.assign(name, value)
            } else {
                false
            }
        }
    }
}

fn issue9925(mut hm: HashMap<String, bool>) {
    let key = "hello".to_string();
    if hm.contains_key(&key) {
        //~^ map_entry
        let bval = hm.get_mut(&key).unwrap();
        *bval = false;
    } else {
        hm.insert(key, true);
    }
}

mod issue9470 {
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct Interner(i32);

    impl Interner {
        const fn new() -> Self {
            Interner(0)
        }

        fn resolve(&self, name: String) -> Option<String> {
            todo!()
        }
    }

    static INTERNER: Mutex<Interner> = Mutex::new(Interner::new());

    struct VM {
        stack: Vec<i32>,
        globals: HashMap<String, i32>,
    }

    impl VM {
        fn stack_top(&self) -> &i32 {
            self.stack.last().unwrap()
        }

        fn resolve(&mut self, name: String, value: i32) -> Result<(), String> {
            if self.globals.contains_key(&name) {
                //~^ map_entry
                self.globals.insert(name, value);
            } else {
                let interner = INTERNER.lock().unwrap();
                return Err(interner.resolve(name).unwrap().to_owned());
            }

            Ok(())
        }
    }
}

fn main() {}
