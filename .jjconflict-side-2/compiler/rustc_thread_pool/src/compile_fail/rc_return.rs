/** ```compile_fail,E0277

use std::rc::Rc;

rustc_thread_pool::join(|| Rc::new(22), || ()); //~ ERROR

``` */
mod left {}

/** ```compile_fail,E0277

use std::rc::Rc;

rustc_thread_pool::join(|| (), || Rc::new(23)); //~ ERROR

``` */
mod right {}
