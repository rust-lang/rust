async fn foo() {}
extern fn foo() {}
const fn foo() {}
const unsafe fn foo() {}
unsafe extern "C" fn foo() {}
unsafe fn foo() {}
async unsafe fn foo() {}
const unsafe fn bar() {}

unsafe trait T {}
auto trait T {}
unsafe auto trait T {}

unsafe impl Foo {}
default impl Foo {}
unsafe default impl Foo {}

unsafe extern "C++" {}
