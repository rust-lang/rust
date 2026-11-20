/*! ```compile_fail,E0277

use std::rc::Rc;

let r = Rc::new(22);
rustc_thread_pool::join(|| r.clone(), || r.clone());
//~^ ERROR

``` */
