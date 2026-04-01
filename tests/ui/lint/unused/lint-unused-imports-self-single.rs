//@ run-rustfix

#![deny(unused_imports)]
#![allow(unreachable_code)]

use std::collections::{HashMap, self as coll};
//~^ ERROR unused import: `HashMap`

use std::io::{self as std_io};
//~^ ERROR unused import: `self as std_io`

use std::sync::{Mutex, self as std_sync};
//~^ ERROR unused import: `self as std_sync`

use std::sync::{mpsc::{self as std_sync_mpsc, Sender}};
//~^ ERROR unused import: `self as std_sync_mpsc`

use std::collections::{hash_map::{self as std_coll_hm, Keys}};
//~^ ERROR unused import: `Keys`

use std::borrow::{self, Cow};
//~^ ERROR unused import: `self`

fn main() {
    let _ = coll::BTreeSet::<String>::default();
    let _ = Mutex::new(String::new());
    let _: Cow<'static, str> = "foo".into();
    let _: Sender<u32> = todo!();
    let _: std_coll_hm::Entry<'static, u32, u32> = todo!();
}
