impl Bar { default const unsafe fn foo() { "hi" } }

impl Baz { default unsafe extern "C" fn foo() { "hi" } }

impl Foo for Bar { default fn foo() { "hi" } }
