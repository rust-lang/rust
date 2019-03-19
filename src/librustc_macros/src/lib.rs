#![feature(proc_macro_hygiene)]
#![deny(rust_2018_idioms)]

use synstructure::decl_derive;

mod hash_stable;

decl_derive!([HashStable, attributes(stable_hasher)] => hash_stable::hash_stable_derive);
