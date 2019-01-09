// compile-flags: -Z continue-parse-after-error

impl ! {} // OK
impl ! where u8: Copy {} // OK

impl Trait Type {} //~ ERROR missing `for` in a trait impl
impl Trait .. {} //~ ERROR missing `for` in a trait impl
impl ?Sized for Type {} //~ ERROR expected a trait, found type
impl ?Sized for .. {} //~ ERROR expected a trait, found type

default unsafe FAIL //~ ERROR expected `impl`, found `FAIL`
