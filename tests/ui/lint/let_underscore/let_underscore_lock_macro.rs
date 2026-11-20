//@ check-fail

 use std::sync::Mutex;

 macro_rules! foo {
     () => {
         let m = Mutex::new(String::from("rusty"));
         let _ = dbg!(m.lock().unwrap()); //~ ERROR: non-binding let on a synchronization lock [let_underscore_lock]
     };
 }

 fn main() {
     let m = Mutex::new(String::from("rusty"));
     let _ = dbg!(m.lock().unwrap()); //~ ERROR: non-binding let on a synchronization lock [let_underscore_lock]

     foo!();
 }
