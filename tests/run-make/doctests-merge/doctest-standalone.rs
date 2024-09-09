#![crate_name = "foo"]
#![crate_type = "lib"]

//! ```standalone
//! foo::init();
//! ```

/// ```standalone
/// foo::init();
/// ```
pub fn init() {
    static mut IS_INIT: bool = false;

    unsafe {
        assert!(!IS_INIT);
        IS_INIT = true;
    }
}
