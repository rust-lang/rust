 #![crate_name = "foo"]
#![feature(staged_api)]
#![stable(feature = "v1", since="1.0.0")]

#[stable(feature = "futures_api", since = "1.36.0")]
//@ has "foo/task/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.36.0"
//@ !has - "//span[@class='sub-heading']/span[@class='since']" "1.0.0"
pub mod task {

    #[doc(inline)]
    #[stable(feature = "futures_api", since = "1.36.0")]
    //@ has "foo/task/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.36.0"
    //@ has "foo/task/ready/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.64.0"
    pub use core::task::*;
}

#[stable(feature = "futures_api", since = "1.36.0")]
//@ has "foo/core/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.36.0"
//@ !has - "//span[@class='sub-heading']/span[@class='since']" "1.0.0"
pub mod core {
    #[stable(feature = "futures_api", since = "1.36.0")]
    //@ has "foo/core/task/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.36.0"
    pub mod task {

        #[stable(feature = "ready_macro", since = "1.64.0")]
        //@ has "foo/core/task/ready/index.html" "//span[@class='sub-heading']/span[@class='since']" "1.64.0"
        pub mod ready {
        }
    }
}
