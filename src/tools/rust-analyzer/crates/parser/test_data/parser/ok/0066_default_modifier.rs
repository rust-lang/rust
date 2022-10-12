trait T {
  default type T = Bar;
  default const f: u8 = 0;
  default fn foo() {}
  default unsafe fn bar() {}
}

impl T for Foo {
  default type T = Bar;
  default const f: u8 = 0;
  default fn foo() {}
  default unsafe fn bar() {}
}

default impl T for () {}
default unsafe impl T for () {}
